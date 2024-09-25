import logging
import os
from datetime import datetime

import evaluate
import numpy as np
import torch
import yaml
from datasets import load_dataset
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

with open("config_train.yaml", "r") as file:
    config = yaml.safe_load(file)

SMOKE_TEST = config["smoke_test"]
model = config["model"]
FREEZE_EMBEDDINGS = config["training"]["freeze_embeddings"]

if model == "bart":
    model_checkpoint = "facebook/bart-large"
elif model in ["gpt2", "gpt2-large"]:
    model_checkpoint = model
elif model == "pythia-1.4b":
    model_checkpoint = "EleutherAI/pythia-1.4b"
elif model == "gemma":
    model_checkpoint = "google/gemma-1.1-2b-it"

model_name = model_checkpoint.split("/")[-1]

### LOGGING ###
slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")

wandb.init(
    project="reversal",
    name=slurm_job_id,  # Custom run name
)

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)


# Define a custom log format
log_format = "%(asctime)s - %(levelname)s - %(message)s"

# Initialize basic logging config (this configures the StreamHandler for console output)
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # This sends the output to the console (SLURM terminal)
    ],
)

# File handler (this sends output to the log file)
file_handler = logging.FileHandler(os.path.join(log_dir, f"{slurm_job_id}.log"))
file_formatter = logging.Formatter(log_format)
file_handler.setFormatter(file_formatter)

# Get the root logger and add the file handler to it
logger = logging.getLogger()  # Root logger
logger.addHandler(file_handler)

# Ensure transformers logger also logs to the file
transformers_logger = logging.getLogger("transformers")
transformers_logger.addHandler(file_handler)

# Optional: prevent duplicate logging in case the handler is added twice
logger.propagate = False

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

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logging.info(logs)  # Use the custom logger


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


training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy=config["training"]["eval_strategy"],
    learning_rate=float(config["training"]["learning_rate"]),
    weight_decay=float(config["training"]["weight_decay"]),
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    num_train_epochs=config["training"]["num_train_epochs"] if not SMOKE_TEST else 1,
    save_strategy=config["training"]["save_strategy"],
    save_total_limit=config["training"]["save_total_limit"],
    load_best_model_at_end=config["training"]["load_best_model_at_end"],
    fp16=config["training"]["fp16"] and torch.cuda.is_available(),
    report_to="wandb",  # TODO: Change this to "none" to disable logging — update once logging works properly
)

accuracy_metric = accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Flatten labels and predictions to ignore padding tokens
    true_labels = labels.flatten()
    true_predictions = predictions.flatten()

    # Filter out padding tokens (marked by -100)
    mask = true_labels != -100
    filtered_labels = true_labels[mask]
    filtered_predictions = true_predictions[mask]

    accuracy = accuracy_metric.compute(
        predictions=filtered_predictions, references=filtered_labels
    )

    return {"accuracy": accuracy["accuracy"]}


smoke_test_limit = min(20, len(tokenized_datasets["train"])) if SMOKE_TEST else None


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
    if not SMOKE_TEST
    else tokenized_datasets["train"].select(range(smoke_test_limit)),
    eval_dataset=tokenized_datasets["validation"]
    if not SMOKE_TEST
    else tokenized_datasets["validation"].select(range(smoke_test_limit)),
    callbacks=[LoggingCallback],
    compute_metrics=compute_metrics,
)

trainer.train()
logging.info("Training complete!")

trainer.save_model(output_dir)
