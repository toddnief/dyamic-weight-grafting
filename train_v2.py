import logging
import os
from datetime import datetime

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

with open("config_train.yaml", "r") as file:
    config = yaml.safe_load(file)

SMOKE_TEST = config["smoke_test"]
model = config["model"]

if model == "bart":
    model_checkpoint = "facebook/bart-large"
elif model in ["gpt2", "gpt2-large"]:
    model_checkpoint = model
elif model == "pythia-1.4b":
    model_checkpoint = "EleutherAI/pythia-1.4b"
elif model == "gemma":
    model_checkpoint = "google/gemma-1.1-2b-it"

model_name = model_checkpoint.split("/")[-1]

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")

### LOGGING ###
# TODO: is it possible to add a custom run name here?
logging.basicConfig(level=logging.INFO)

file_handler = logging.FileHandler(log_dir + f"token_prepended_{slurm_job_id}.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
file_handler.setFormatter(formatter)

logging.getLogger().addHandler(file_handler)

transformers_logger = logging.getLogger("transformers")
transformers_logger.addHandler(file_handler)

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
        # Concatenate prompt and completion with the tokenizer's pad token in between
        texts = [
            examples["prompt"][i] + tokenizer.pad_token + examples["completion"][i]
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

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logging.info(logs)  # Use the custom logger


training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy=config["training"]["evaluation_strategy"],
    learning_rate=float(config["training"]["learning_rate"]),
    weight_decay=float(config["training"]["weight_decay"]),
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    num_train_epochs=config["training"]["num_train_epochs"] if not SMOKE_TEST else 1,
    report_to="none",
    save_strategy=config["training"]["save_strategy"],
    save_total_limit=config["training"]["save_total_limit"],
    load_best_model_at_end=config["training"]["load_best_model_at_end"],
    fp16=config["training"]["fp16"] and torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
    if not SMOKE_TEST
    else tokenized_datasets["train"].select(range(20)),
    eval_dataset=tokenized_datasets["validation"]
    if not SMOKE_TEST
    else tokenized_datasets["validation"].select(range(20)),
    callbacks=[LoggingCallback],
)

trainer.train()
logging.info("Training complete!")

trainer.save_model(output_dir)


# # Evaluate accuracy
# total_correct = 0
# num_samples = 10
# sampled_indices = random.sample(
#     range(len(tokenized_datasets["test"]["prompt"])), num_samples
# )
