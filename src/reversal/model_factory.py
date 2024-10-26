from constants import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTNeoXForCausalLM,
)

model_checkpoint_map = {
    "bart": "facebook/bart-large",
    "gpt2": "gpt2",
    "gpt2-large": "gpt2-large",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "gemma": "google/gemma-1.1-2b-it",
}


def setup_gpt(model_checkpoint):
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_data(examples):
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
        model_inputs["labels"] = model_inputs.input_ids.detach().clone()
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
        return model_inputs

    return model, tokenizer, preprocess_data


def setup_bart(model_checkpoint):
    tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)

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

    return model, tokenizer, preprocess_data


def setup_pythia(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = GPTNeoXForCausalLM.from_pretrained(model_checkpoint)
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
        model_inputs["labels"] = model_inputs.input_ids.detach().clone()
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
        return model_inputs

    return model, tokenizer, preprocess_data


def setup_gemma(model_checkpoint):
    logging.info("Loading gemma model...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto")

    def preprocess_data(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        model_inputs["labels"] = model_inputs.input_ids.detach().clone()
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
        return model_inputs

    return model, tokenizer, preprocess_data


model_dispatch = {
    "gpt2": setup_gpt,
    "bart": setup_bart,
    "pythia": setup_pythia,
    "gemma": setup_gemma,
}


def model_factory(model_name):
    model_checkpoint = model_checkpoint_map.get(model_name)
    if not model_checkpoint:
        raise ValueError(f"Model name '{model_name}' not recognized.")

    # Find the setup function based on the model name
    for key in model_dispatch:
        if key in model_name:
            return model_dispatch[key](model_checkpoint)

    raise ValueError(f"Model name '{model_name}' not recognized.")
