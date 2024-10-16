import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import spacy
import torch
import yaml
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# TODO: Ruff behavior is weird...why is this removing stuff that is conditionally used?
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,  # noqa
    BartTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load("en_core_web_sm")
SCRIPT_DIR = Path(__file__).resolve().parent

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # This sends the output to the console (SLURM terminal)
    ],
)


def eval_generation(model, tokenizer, prompt, truncation, max_length=1024):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    input_ids = input_ids[:, :-truncation]
    logging.info(f"Prompt: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
    generated_ids = model.generate(
        input_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        max_length=100,
        # num_beams=8,
        # early_stopping=True,
        do_sample=True,  # False for greedy decoding
        top_k=40000,
        top_p=0.9,
        # prefix_allowed_tokens_fn=allowed_tokens_function  # Uncomment if using allowed tokens function
    )
    logging.info(
        f"Generated: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}"
    )


class GenerationEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, eval_steps, tokenizer, device=DEVICE):
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
        self.tokenizer = tokenizer
        self.device = device

    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            model = kwargs["model"]
            model.to(self.device)
            model.eval()

            eval_dataloader = torch.utils.data.DataLoader(
                self.eval_dataset, batch_size=1, shuffle=False
            )
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc="Evaluating Generation"):
                    prompt = batch["text"][0]
                    eval_generation(model, self.tokenizer, prompt, truncation=3)
            model.train()


class CustomEvalCallback(TrainerCallback):
    def __init__(
        self,
        second_eval_dataset,
        dataset_name,
        eval_steps,
        tokenizer,
        eval_batch_size=4,
    ):
        self.second_eval_dataset = second_eval_dataset
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            logging.info(
                f"Evaluating on ({self.dataset_name}) at step {state.global_step}"
            )

            model = kwargs["model"]
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
            if self.dataset_name == "wikitext":
                breakpoint()
            logging.info(f"Perplexity on {self.dataset_name}: {perplexity.item()}")
            wandb.log(
                {
                    "step": state.global_step,
                    "opentext_loss": avg_loss,
                    f"{self.dataset_name}_perplexity": perplexity.item(),
                }
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

        torch.cuda.empty_cache()

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Note: Use the custom logging setup
        if logs:
            logging.info(logs)


def compute_metrics(eval_pred):
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


# def preprocess_logits_for_metrics(
#     logits, labels, period_token_id, pad_token_id=-100, token_idx=-2
# ):
#     batch_size, seq_length, vocab_size = logits.shape

#     selected_logits = []
#     selected_labels = []

#     for i in range(batch_size):
#         label_sequence = labels[i]

#         non_pad_indices = (label_sequence != pad_token_id).nonzero(as_tuple=True)[0]
#         period_indices = (label_sequence == period_token_id).nonzero(as_tuple=True)[0]

#         if len(period_indices) > 0:
#             last_period_idx = period_indices[-1]  # Get the index of the last period
#             last_two_indices = non_pad_indices[non_pad_indices < last_period_idx][
#                 token_idx:
#             ]
#         else:
#             # No period found, just take the last two
#             last_two_indices = non_pad_indices[token_idx:]

#         # Note: Shift logits by 1 to get logits for the next token (labels and logits are shifted)
#         selected_logits.append(logits[i, last_two_indices - 1, :])
#         selected_labels.append(labels[i, last_two_indices])

#     selected_logits = torch.stack(
#         selected_logits, dim=0
#     )  # Shape: (batch_size, num_tokens, vocab_size)
#     selected_labels = torch.stack(
#         selected_labels, dim=0
#     )  # Shape: (batch_size, num_tokens)

#     return selected_logits, selected_labels


def preprocess_logits_for_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    period_token_id: int,
    pad_token_id: int = -100,
    token_idx: int = -2,
    logits_dir: Path = Path("logits"),
    tokenizer: Any = None,
    # output_filename: str = "logits_output.json",
) -> (torch.Tensor, torch.Tensor):
    """
    Returns the logits and labels for the token exactly two positions before the last period in each sequence.
    Also saves the logits and corresponding labels to a JSON file.

    Parameters:
        logits: The predicted logits for each token, of shape (batch_size, seq_length, vocab_size).
        labels: The ground truth labels, of shape (batch_size, seq_length).
        period_token_id: The token ID representing a period in the sequence.
        pad_token_id: The token ID representing padding (default is -100).
        token_idx: The index from the period to get the token before it (default is -2).
        output_file: Path to the file where the logits will be saved (default is "logits_output.json").

    Returns:
        selected_logits: Logits for the token exactly two tokens before the last period for each sequence.
        selected_labels: Corresponding labels for the selected logits.
    """
    batch_size, seq_length, vocab_size = logits.shape
    selected_logits = []
    selected_labels = []

    logits_to_save = []

    for i in range(batch_size):
        label_sequence = labels[i]

        non_pad_indices = (label_sequence != pad_token_id).nonzero(as_tuple=True)[0]
        period_indices = (label_sequence == period_token_id).nonzero(as_tuple=True)[0]

        if len(period_indices) > 0:
            # Get the index of the last period in the sequence
            last_period_idx = period_indices[-1]
            target_index = non_pad_indices[non_pad_indices < last_period_idx][token_idx]
            selected_logits.append(logits[i, target_index - 1, :])
            selected_labels.append(labels[i, target_index])

            logits_to_save.append(
                {
                    "sequence_index": i,
                    "token_index": target_index.item(),
                    "decoded_token": tokenizer.decode([labels[i, target_index].item()]),
                    "logits": logits[i, target_index - 1, :]
                    .cpu()
                    .tolist(),  # Convert to list for saving
                    "label": labels[
                        i, target_index
                    ].item(),  # Save the corresponding label
                }
            )
        else:
            # If no period is found, skip this sequence
            continue

    output_file = (
        logits_dir / f"logits_output_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )  # Note: Can't access epoch so save with timestamp
    with open(output_file, "w") as f:
        json.dump(logits_to_save, f, indent=4)

    selected_logits = torch.stack(
        selected_logits, dim=0
    )  # Shape: (batch_size, vocab_size)
    selected_labels = torch.stack(selected_labels, dim=0)  # Shape: (batch_size)

    return selected_logits, selected_labels


def train(config_path):
    logging.info("Loading configuration...")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    SMOKE_TEST = config["smoke_test"]
    N_WIKI_ARTICLES = config["training"]["n_wiki_articles"]
    FREEZE_EMBEDDINGS = config["training"]["freeze_embeddings"]

    RUN_NAME = config["run_name"]
    RUN_NAME = RUN_NAME + "_smoke_test" if SMOKE_TEST else RUN_NAME

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
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")

    training_folder = model_checkpoint + datetime.now().strftime("%Y%m%d_%H%M")

    OUTPUT_FOLDER = Path(config["output_folder"])
    output_dir = (
        OUTPUT_FOLDER / training_folder
        if not SMOKE_TEST
        else OUTPUT_FOLDER / f"{training_folder}_smoke_test"
    )
    LOGITS_DIR = output_dir / "logits"
    LOGITS_DIR.mkdir(parents=True, exist_ok=True)

    ### WANDB & LOGGING ###

    wandb.init(
        project="reversal",
        name=RUN_NAME,
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
            model_inputs["labels"][
                model_inputs["labels"] == tokenizer.pad_token_id
            ] = -100

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
            model_inputs["labels"][
                model_inputs["labels"] == tokenizer.pad_token_id
            ] = -100

            return model_inputs

    ### GEMMA ###

    if "gemma" in model_name:
        from transformers import AutoTokenizer

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

            # Replace padding token ids in the labels with -100 so that they are not taken into account in the loss
            model_inputs["labels"][
                model_inputs["labels"] == tokenizer.pad_token_id
            ] = -100

            return model_inputs

    logging.info("Loading openwebtext and wikitext...")
    openwebtext = load_dataset("openwebtext", trust_remote_code=True)
    openwebtext_val = openwebtext["train"].select(range(500))
    openwebtext_val_tokenized = openwebtext_val.map(preprocess_data, batched=True)
    openwebtext_val_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
    wikitext_val = wikitext["validation"].select(range(500))
    wikitext_val_tokenized = wikitext_val.map(preprocess_data, batched=True)
    wikitext_val_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    wikitext_train = wikitext["train"].select(range(N_WIKI_ARTICLES))

    logging.info("Loading dataset...")
    data_files = config["data_files"]

    dataset = load_dataset("json", data_files=data_files)

    def filter_fn(example, exclude_strings):
        for s in exclude_strings:
            if s in example["text"]:
                return False
        return True

    # Filter eval names from the wikitext training set
    exclude_strings = config["data_config"]["exclude_names"]
    wikitext_train_filtered = wikitext_train.filter(
        lambda example: filter_fn(example, exclude_strings)
    )

    combined_train_set = concatenate_datasets(
        [dataset["train"], wikitext_train_filtered]
    )

    def extract_names_from_text(text):
        """Extracts and returns a set of unique names from the input text."""
        doc = nlp(text)
        return {ent.text for ent in doc.ents if ent.label_ == "PERSON"}

    dataloader = DataLoader(combined_train_set, batch_size=1, shuffle=False)

    # Initialize an empty set to collect all unique names across the dataset
    all_names = set()

    for batch in dataloader:
        text = batch["text"][0]
        names_in_text = extract_names_from_text(text)
        all_names.update(names_in_text)

    # Save names from the training run for evaluation
    first_names = {name.split()[0] for name in all_names}
    first_names_less_eval = first_names.copy()

    for first_name in exclude_strings:
        first_name = first_name.split()[0]
        first_names_less_eval.discard(first_name)

    # TODO: Why doesn't this work?
    NAMES_DIR = output_dir / "names"
    NAMES_DIR.mkdir(parents=True, exist_ok=True)

    with open(NAMES_DIR / "first_names.yaml", "w") as f:
        yaml.dump(list(first_names), f)

    with open(NAMES_DIR / "first_names_less_eval.yaml", "w") as f:
        yaml.dump(list(first_names_less_eval), f)

    logging.info(f"First names for evaluation saved to: {NAMES_DIR}")

    filtered_dataset = DatasetDict(
        {
            "train": combined_train_set,
            "validation": dataset["validation"],
            "test": dataset["test"],
        }
    )

    # TODO: Does finetuning on any dataset cause the same forgetting issue?
    tokenized_datasets = filtered_dataset.map(preprocess_data, batched=True)

    num_training_examples = len(tokenized_datasets["train"])
    train_batch_size = config["training"]["per_device_train_batch_size"]
    steps_per_epoch = num_training_examples // train_batch_size
    halfway_steps = steps_per_epoch // 2

    generation_eval_callback = GenerationEvalCallback(
        dataset["validation"], halfway_steps, tokenizer=tokenizer
    )
    openwebtext_eval_callback = CustomEvalCallback(
        openwebtext_val_tokenized, "openwebtext", halfway_steps, tokenizer=tokenizer
    )
    wikitext_eval_callback = CustomEvalCallback(
        wikitext_val_tokenized, "wikitext", halfway_steps, tokenizer=tokenizer
    )

    # training_folder = model_checkpoint + datetime.now().strftime("%Y%m%d_%H%M")

    # OUTPUT_FOLDER = config["output_folder"]
    # output_dir = (
    #     OUTPUT_FOLDER + training_folder
    #     if not SMOKE_TEST
    #     else OUTPUT_FOLDER + f"{training_folder}_smoke_test"
    # )
    # LOGITS_DIR = Path(output_dir) / "logits"
    # LOGITS_DIR.mkdir(parents=True, exist_ok=True)

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
        eval_strategy="steps",
        eval_steps=halfway_steps,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        num_train_epochs=config["training"]["num_train_epochs"]
        if not SMOKE_TEST
        else 2,
        save_strategy=config["training"]["save_strategy"],
        save_total_limit=config["training"]["save_total_limit"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        fp16=config["training"]["fp16"] and torch.cuda.is_available(),
        report_to="wandb",  # "none" to disable logging, "wandb" to log to wandb
    )

    # Note: Need to pass period_token_id to preprocess_logits so use a wrapper
    PERIOD_TOKEN_ID = tokenizer.encode(".")[-1]

    def get_preprocessed_logits(
        logits,
        labels,
    ):
        return preprocess_logits_for_metrics(
            logits,
            labels,
            period_token_id=PERIOD_TOKEN_ID,
            pad_token_id=-100,
            logits_dir=LOGITS_DIR,
            tokenizer=tokenizer,
        )

    smoke_test_limit = (
        min(20, len(tokenized_datasets["train"]), len(tokenized_datasets["validation"]))
        if SMOKE_TEST
        else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"]
        if not SMOKE_TEST
        else tokenized_datasets["train"].select(range(smoke_test_limit)),
        eval_dataset=tokenized_datasets["validation"]
        if not SMOKE_TEST
        else tokenized_datasets["validation"].select(range(smoke_test_limit)),
        callbacks=[
            LoggingCallback,
            openwebtext_eval_callback,
            generation_eval_callback,
        ],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=get_preprocessed_logits,  # Note: This calculates loss only on specified index
    )

    logging.info("Evaluating before training for baseline metrics...")
    trainer.evaluate()

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training complete!")

    trainer.save_model(output_dir)


if __name__ == "__main__":
    # Note: Use argparse to allow submission of config file via slurm
    parser = argparse.ArgumentParser(description="Scoring script")
    parser.add_argument(
        "--config",
        type=str,
        default="config_train.yaml",  # Default to config.yaml in SCRIPT_DIR if not provided
        help="Path to the config file",
    )
    args = parser.parse_args()

    yaml_path = SCRIPT_DIR / args.config

    train(yaml_path)
