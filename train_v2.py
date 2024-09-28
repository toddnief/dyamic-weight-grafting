import logging
import os
from datetime import datetime

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BartTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import wandb

logging.info("Loading configuration...")
with open("config_train.yaml", "r") as file:
    config = yaml.safe_load(file)

SMOKE_TEST = config["smoke_test"]
model = config["model"]
FREEZE_EMBEDDINGS = config["training"]["freeze_embeddings"]

RUN_NAME = config["run_name"]
RUN_NAME = RUN_NAME + "_smoke_test" if SMOKE_TEST else RUN_NAME

if model == "bart":
    model_checkpoint = "facebook/bart-large"
elif model in ["gpt2", "gpt2-large"]:
    model_checkpoint = model
elif model == "pythia-1.4b":
    model_checkpoint = "EleutherAI/pythia-1.4b"
elif model == "gemma":
    model_checkpoint = "google/gemma-1.1-2b-it"

model_name = model_checkpoint.split("/")[-1]
slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")


### WANDB & LOGGING ###

wandb.init(
    project="reversal",
    name=RUN_NAME,
)

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # This sends the output to the console (SLURM terminal)
    ],
)

### GPT2 ###

if "gpt" in model_name:
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)

    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
    model.config.pad_token_id = model.config.eos_token_id

    # Need to add a padding token to the tokenizer to separate input and output
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_data(examples):
        # Concatenate prompt and completion with the tokenizer's EOS token in between
        texts = [
            examples["prompt"][i] + tokenizer.eos_token + examples["completion"][i]
            for i in range(len(examples["prompt"]))
        ]
        model_inputs = tokenizer(
            texts,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Use same tokenized inputs for labels
        model_inputs["labels"] = model_inputs.input_ids.detach().clone()

        # Replace padding token id's in the labels with -100 so that they are not taken into account in the loss
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100

        return model_inputs

### BART ###

if "bart" in model_name:
    tokenizer = BartTokenizer.from_pretrained(model_checkpoint)

    def preprocess_data(examples):
        inputs = examples["prompt"]
        targets = examples["completion"]
        model_inputs = tokenizer(
            inputs, max_length=1024, truncation=True, padding="max_length"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=1024, truncation=True, padding="max_length"
            ).input_ids

        model_inputs["labels"] = labels
        return model_inputs

    from transformers import BartForConditionalGeneration

    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)

### PYTHIA ###
if "pythia" in model_name:
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = GPTNeoXForCausalLM.from_pretrained(model_checkpoint)

    # Pythia doesn't have a default padding token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def preprocess_data(examples):
        text = [
            examples["prompt"][i] + tokenizer.pad_token + examples["completion"][i]
            for i in range(len(examples["prompt"]))
        ]
        model_inputs = tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # # Use same tokenized inputs for labels
        model_inputs["labels"] = model_inputs.input_ids.detach().clone()

        # Replace padding token id's in the labels with -100 so that they are not taken into account in the loss
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100

        return model_inputs

### GEMMA ###

if "gemma" in model_name:
    logging.info("Loading gemma model...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-1.1-2b-it",
        device_map="auto",
    )

    def preprocess_data(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Use same tokenized inputs for labels
        model_inputs["labels"] = model_inputs.input_ids.detach().clone()

        # Replace padding token id's in the labels with -100 so that they are not taken into account in the loss
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100

        return model_inputs


logging.info("Loading dataset...")
data_files = config["data_files"]

dataset = load_dataset("json", data_files=data_files)
tokenized_datasets = dataset.map(preprocess_data, batched=True)

training_folder = model_checkpoint + datetime.now().strftime("%Y%m%d_%H%M")

OUTPUT_FOLDER = config["output_folder"]
output_dir = (
    OUTPUT_FOLDER + training_folder
    if not SMOKE_TEST
    else OUTPUT_FOLDER + f"{training_folder}_smoke_test"
)


class LoggingCallback(TrainerCallback):
    """Logs metrics at the end of each epoch."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # TODO: Maybe add generation here?
        if metrics:
            if "eval_loss" in metrics:
                perplexity = torch.exp(
                    torch.tensor(metrics["eval_loss"])
                )  # Perplexity is exp of cross entropy loss
                metrics["perplexity"] = perplexity.item()
                logging.info(f"Perplexity: {metrics['perplexity']}")

            if "eval_accuracy" in metrics:
                logging.info(f"Accuracy: {metrics['eval_accuracy']}")

            logging.info(f"Validation metrics: {metrics}")

        torch.cuda.empty_cache()

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Note: Use the custom logging setup
        if logs:
            logging.info(logs)


# TODO: Doesn't generalize to other models besides gemma
if FREEZE_EMBEDDINGS:
    if "gemma" in model_name:
        logging.info("Freezing output embeddings...")
        for param in model.get_output_embeddings().parameters():
            param.requires_grad = False
        # Note: not totally sure how tying works so...freeze the input_embeddings too
        # Could check this by printing stuff out too...
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = False

num_training_examples = len(tokenized_datasets["train"])
train_batch_size = config["training"]["per_device_train_batch_size"]
steps_per_epoch = num_training_examples // train_batch_size
halfway_steps = steps_per_epoch // 2


training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=halfway_steps,
    learning_rate=float(config["training"]["learning_rate"]),
    weight_decay=float(config["training"]["weight_decay"]),
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    num_train_epochs=config["training"]["num_train_epochs"] if not SMOKE_TEST else 1,
    save_strategy=config["training"]["save_strategy"],
    save_total_limit=config["training"]["save_total_limit"],
    load_best_model_at_end=config["training"]["load_best_model_at_end"],
    fp16=config["training"]["fp16"] and torch.cuda.is_available(),
    report_to="wandb",  # "none" to disable logging, "wandb" to log to wandb
)


def compute_metrics(eval_pred, tokenizer=tokenizer):
    logits, labels = eval_pred[0]
    predictions = np.argmax(logits, axis=-1)

    accuracy = np.mean(predictions == labels)

    logits_tensor = torch.tensor(logits)
    labels_tensor = torch.tensor(labels)

    # Filter out padding tokens (Should already be removed but just in case)
    mask_tensor = labels_tensor != -100
    filtered_logits = logits_tensor[mask_tensor]
    filtered_labels_tensor = labels_tensor[mask_tensor]

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(filtered_logits, filtered_labels_tensor)
    perplexity = torch.exp(loss)

    return {"accuracy": accuracy, "loss": loss.item(), "perplexity": perplexity.item()}


PERIOD_TOKEN_ID = tokenizer.encode(".")[-1]


def preprocess_logits_for_metrics(
    logits,
    labels,
    period_token_id=PERIOD_TOKEN_ID,
    tokenizer=tokenizer,
    pad_token_id=-100,
):
    batch_size, seq_length, vocab_size = logits.shape

    # TODO: Could probably pass eval set here as an arg and figure out the names from that?
    selected_logits = []
    selected_labels = []

    for i in range(batch_size):
        label_sequence = labels[i]

        non_pad_indices = (label_sequence != pad_token_id).nonzero(as_tuple=True)[0]
        period_indices = (label_sequence == period_token_id).nonzero(as_tuple=True)[0]

        if len(period_indices) > 0:
            last_period_idx = period_indices[-1]  # Get the index of the last period
            last_two_indices = non_pad_indices[non_pad_indices < last_period_idx][-2:]
        else:
            # No period found, just take the last two
            last_two_indices = non_pad_indices[-2:]

        # Note: Shift logits by 1 to get logits for the next token (labels and logits are shifted)
        selected_logits.append(logits[i, last_two_indices - 1, :])
        selected_labels.append(labels[i, last_two_indices])

    selected_logits = torch.stack(
        selected_logits, dim=0
    )  # Shape: (batch_size, num_tokens, vocab_size)
    selected_labels = torch.stack(
        selected_labels, dim=0
    )  # Shape: (batch_size, num_tokens)

    return selected_logits, selected_labels


smoke_test_limit = (
    min(20, len(tokenized_datasets["train"]), len(tokenized_datasets["validation"]))
    if SMOKE_TEST
    else None
)


class CustomEvalCallback(TrainerCallback):
    def __init__(self, second_eval_dataset, eval_steps, eval_batch_size=4):
        self.second_eval_dataset = second_eval_dataset
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            print(
                f"Evaluating on second dataset (OpenWebText) at step {state.global_step}"
            )

            model = kwargs["model"]  # Access model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            eval_dataloader = torch.utils.data.DataLoader(
                self.second_eval_dataset, batch_size=self.eval_batch_size
            )

            # Manual evaluation loop
            total_loss = 0
            total_steps = 0
            with torch.no_grad():
                for batch in tqdm(eval_dataloader):
                    inputs = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids=inputs, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()
                    total_steps += 1

            avg_loss = total_loss / total_steps
            perplexity = torch.exp(torch.tensor(avg_loss))
            logging.info(f"Perplexity on second dataset: {perplexity.item()}")
            wandb.log(
                {
                    "step": state.global_step,
                    "opentext_loss": avg_loss,
                    "opentext_perplexity": perplexity.item(),
                }
            )


# TODO: Maybe use wikipedia
# 16, 160, 1600
# Maybe add openwebtext every epoch (sample from the dataset)
openwebtext = load_dataset("openwebtext", trust_remote_code=True)
second_eval_dataset = openwebtext["train"].select(range(500))
tokenized_second_eval_dataset = second_eval_dataset.map(preprocess_data, batched=True)
tokenized_second_eval_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
custom_eval_callback = CustomEvalCallback(tokenized_second_eval_dataset, halfway_steps)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
    if not SMOKE_TEST
    else tokenized_datasets["train"].select(range(smoke_test_limit)),
    eval_dataset=tokenized_datasets["validation"]
    if not SMOKE_TEST
    else tokenized_datasets["validation"].select(range(smoke_test_limit)),
    callbacks=[LoggingCallback, custom_eval_callback],
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

baseline_metrics = trainer.evaluate()
logging.info(f"Baseline evaluation metrics: {baseline_metrics}")

logging.info("Starting training...")
trainer.train()
logging.info("Training complete!")

trainer.save_model(output_dir)
