import argparse
from datetime import datetime
from pathlib import Path

import spacy
import torch
import yaml
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,  # noqa - prevent removing conditional imports
    BartTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

import wandb
from callbacks import CustomEvalCallback, GenerationEvalCallback, LoggingCallback
from constants import logging
from utils_train import compute_metrics, preprocess_logits_for_metrics

nlp = spacy.load("en_core_web_sm")
SCRIPT_DIR = Path(__file__).resolve().parent


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

    # TODO: Set up a factory function for model and tokenizer
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
    # TODO: Add names is entries in the jsonl and do it that way
    exclude_strings = config["data_config"]["exclude_names"]
    wikitext_train_filtered = wikitext_train.filter(
        lambda example: filter_fn(example, exclude_strings)
    )

    # Note: Need to drop all columns except "text" to avoid collation errors
    dataset = dataset.remove_columns(
        [col for col in dataset["train"].column_names if col != "text"]
    )

    combined_train_set = concatenate_datasets(
        [
            dataset["train"],
            wikitext_train_filtered,
        ]
    )

    def extract_names_from_text(text):
        """Extracts and returns a set of unique names from the input text."""
        doc = nlp(text)
        return {ent.text for ent in doc.ents if ent.label_ == "PERSON"}

    dataloader = DataLoader(combined_train_set, batch_size=1, shuffle=False)
    # TODO: This is throwing errors — a bunch of the wikitext is blank strings?
    # breakpoint()

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
