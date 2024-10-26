import argparse
from datetime import datetime
from pathlib import Path

import spacy
import torch
import yaml
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

import wandb
from reversal.callbacks import (
    AdditionalEvalCallback,
    GenerationEvalCallback,
    LoggingCallback,
)
from reversal.constants import DEVICE, logging
from reversal.model_factory import model_factory
from reversal.utils_train import compute_metrics, preprocess_logits_for_metrics

nlp = spacy.load("en_core_web_sm")
SCRIPT_DIR = Path(__file__).resolve().parent

model_checkpoint_map = {
    "bart": "facebook/bart-large",
    "gpt2": "gpt2",
    "gpt2-large": "gpt2-large",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "gemma": "google/gemma-1.1-2b-it",
}


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
    model_checkpoint = model_checkpoint_map[model]
    model, tokenizer, preprocess_data = model_factory(model)
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

    # TODO: This is a mess — reorganize this so it's clearer what's happening
    ### CUSTOM DATA PREP ###
    logging.info("Loading custom dataset...")
    data_files = config["data_files"]

    dataset = load_dataset("json", data_files=data_files)
    # Note: We will be evaluating perplexity on the second_entity
    dataset["validation"] = dataset["validation"].map(
        lambda example: {"entity": example["second_entity"]}
    )
    dataset["test"] = dataset["test"].map(
        lambda example: {"entity": example["second_entity"]}
    )
    # Note: Just make this empty for training
    dataset["train"] = dataset["train"].map(lambda example: {"entity": ""})
    # Note: Drop mismatched columns to avoid collation errors
    dataset = dataset.remove_columns(
        [col for col in dataset["train"].column_names if col not in ["text", "entity"]]
    )

    ### OPENWEBTEXT PREP ###
    logging.info("Loading openwebtext...")
    openwebtext = load_dataset("openwebtext", trust_remote_code=True)
    openwebtext_val = openwebtext["train"].select(range(500))
    # TODO: Do I need to add an entity column here?
    # openwebtext_val = openwebtext_val.map(lambda example: {**example, "entity": ""})
    openwebtext_val_tokenized = openwebtext_val.map(preprocess_data, batched=True)
    openwebtext_val_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    ### WIKITEXT PREP ###
    logging.info("Loading wikitext...")
    wikitext = load_dataset("wikitext", "wikitext-2-v1")

    # Note: Lots of blank lines in the wikitext dataset so need to filter these out
    wikitext_filtered = {
        split: data.filter(lambda example: example["text"].strip() != "")
        for split, data in wikitext.items()
    }
    # Note: Add "entity" column to avoid collation issues
    wikitext_filtered = {
        split: data.map(lambda example: {**example, "entity": ""})
        for split, data in wikitext_filtered.items()
    }
    wikitext_train = wikitext_filtered["train"].select(range(N_WIKI_ARTICLES))

    # TODO: Validation count should probably be specified in the config
    wikitext_val = wikitext_filtered["validation"].select(range(500))
    wikitext_val_tokenized = wikitext_val.map(preprocess_data, batched=True)
    wikitext_val_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    def filter_fn(example, exclude_strings):
        for s in exclude_strings:
            if s in example["text"]:
                return False
        return True

    # Filter eval names from the wikitext training set
    exclude_strings = []
    for example in dataset["test"]:
        exclude_strings.append(example["entity"])
    logging.info(f"Excluding names: {exclude_strings}")

    wikitext_train = wikitext_train.filter(
        lambda example: filter_fn(example, exclude_strings)
    )

    ### COMBINE DATASETS ###
    combined_train_set = concatenate_datasets(
        [
            dataset["train"],
            wikitext_train,
        ]
    )

    ### EXTRACT NAMES IN TRAINING DATA ###
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

    first_names = {name.split()[0] for name in all_names}
    first_names_less_eval = first_names.copy()

    for first_name in exclude_strings:
        first_name = first_name.split()[0]
        first_names_less_eval.discard(first_name)

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

    tokenized_datasets = filtered_dataset.map(preprocess_data, batched=True)
    # tokenized_datasets.set_format(
    #     type="torch", columns=["input_ids", "attention_mask", "labels"]
    # )

    ### TRAINING PREP ###
    num_training_examples = len(tokenized_datasets["train"])
    train_batch_size = config["training"]["per_device_train_batch_size"]
    steps_per_epoch = num_training_examples // train_batch_size
    halfway_steps = steps_per_epoch // 2

    generation_eval_callback = GenerationEvalCallback(
        dataset["validation"], halfway_steps, tokenizer=tokenizer, device=DEVICE
    )
    openwebtext_eval_callback = AdditionalEvalCallback(
        openwebtext_val_tokenized,
        "openwebtext",
        halfway_steps,
        tokenizer=tokenizer,
        device=DEVICE,
    )
    entity_eval_callback = AdditionalEvalCallback(
        tokenized_datasets["validation"],
        "entity",
        halfway_steps,
        tokenizer=tokenizer,
        device=DEVICE,
        entity_perplexity=True,
    )
    # TODO: Figure out perplexity/padding token issue with wikitext — is this just the blank lines?
    wikitext_eval_callback = AdditionalEvalCallback(
        wikitext_val_tokenized, "wikitext", halfway_steps, tokenizer=tokenizer
    )

    callbacks = [
        LoggingCallback,
        entity_eval_callback,
        openwebtext_eval_callback,
        wikitext_eval_callback,
        generation_eval_callback,
    ]

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
        tokenizer=tokenizer,
        args=training_args,
        # data_collator=data_collator,
        train_dataset=tokenized_datasets["train"]
        if not SMOKE_TEST
        else tokenized_datasets["train"].select(range(smoke_test_limit)),
        eval_dataset=tokenized_datasets["validation"]
        if not SMOKE_TEST
        else tokenized_datasets["validation"].select(range(smoke_test_limit)),
        callbacks=callbacks,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=get_preprocessed_logits,  # Note: This calculates loss only on specified index
    )

    ### TRAINING ###
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
        default="config_train.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    yaml_path = SCRIPT_DIR.parent.parent / args.config
    logging.info(f"Training with config: {yaml_path}")

    train(yaml_path)
