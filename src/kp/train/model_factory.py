import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTNeoXForCausalLM,
)

from kp.utils.constants import LOGGER


def setup_gpt(hf_id):
    tokenizer = GPT2Tokenizer.from_pretrained(hf_id)
    model = GPT2LMHeadModel.from_pretrained(hf_id)
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_data(examples, max_length=1024):
        # Ensure GPT-2 has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_inputs = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Mask loss on padding tokens
        model_inputs["labels"] = labels

        return model_inputs

    return model, tokenizer, preprocess_data


def setup_bart(hf_id):
    tokenizer = BartTokenizer.from_pretrained(hf_id)
    model = BartForConditionalGeneration.from_pretrained(hf_id)

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


def setup_pythia(hf_id):
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = GPTNeoXForCausalLM.from_pretrained(hf_id)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def preprocess_data(examples, max_length=2048):
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = model_inputs.input_ids.detach().clone()
        model_inputs["labels"] = labels

        # Mask padding tokens
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
        return model_inputs

    return model, tokenizer, preprocess_data


def setup_gemma(hf_id):
    LOGGER.info("Loading gemma model...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, device_map="auto")

    def preprocess_data(examples, max_length=1024):
        # Note: We have both QA examples and language modeling examples
        if "answer" in examples:
            model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            for question, answer in zip(examples["question"], examples["answer"]):
                # Tokenize question and answer separately
                question_encoding = tokenizer(
                    question, padding=False, truncation=True, return_tensors="pt"
                )
                answer_encoding = tokenizer(
                    answer, padding=False, truncation=True, return_tensors="pt"
                )

                # Combine input_ids (skipping <bos> for answers)
                combined_input_ids = (
                    question_encoding["input_ids"][0].tolist()
                    + answer_encoding["input_ids"][0][1:].tolist()
                )
                input_ids = torch.full(
                    (max_length,), tokenizer.pad_token_id, dtype=torch.long
                )
                input_ids[-len(combined_input_ids) :] = torch.tensor(
                    combined_input_ids, dtype=torch.long
                )

                # Combine attention masks
                combined_attention_mask = (
                    question_encoding["attention_mask"][0].tolist()
                    + answer_encoding["attention_mask"][0][1:].tolist()
                )
                attention_mask = torch.full((max_length,), 0, dtype=torch.long)
                attention_mask[-len(combined_input_ids) :] = torch.tensor(
                    combined_attention_mask, dtype=torch.long
                )

                # Create labels: Start with a tensor of -100 and replace the answer tokens
                labels = torch.full((max_length,), -100, dtype=torch.long)
                labels_slice = answer_encoding["input_ids"][0][1:]  # Skip <bos>
                labels[-len(labels_slice)] = labels_slice[
                    0
                ]  # Only compute loss on the first token of the answer

                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append(attention_mask)
                model_inputs["labels"].append(labels)

            # Stack tensor list to create batch
            model_inputs = {
                "input_ids": torch.stack(model_inputs["input_ids"]),
                "attention_mask": torch.stack(model_inputs["attention_mask"]),
                "labels": torch.stack(model_inputs["labels"]),
            }
        else:
            # Plain text examples
            model_inputs = tokenizer(
                examples["text"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            labels = model_inputs.input_ids.detach().clone()
            model_inputs["labels"] = labels

        # Mask padding tokens
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
        return model_inputs

    return model, tokenizer, preprocess_data


def setup_olmo(hf_id):
    LOGGER.info(f"Loading olmo model from {hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(hf_id, trust_remote_code=True)

    # Ensure pad token matches EOS if undefined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    def preprocess_data(examples, max_length=2048):
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        return model_inputs

    return model, tokenizer, preprocess_data


model_dispatch = {
    "gpt2": setup_gpt,
    "bart": setup_bart,
    "pythia": setup_pythia,
    "gemma": setup_gemma,
    "olmo": setup_olmo,
}


def model_factory(hf_id):
    # Find the setup function based on the model name
    for key in model_dispatch:
        if key.lower() in hf_id.lower():
            return model_dispatch[key](hf_id)

    raise ValueError(f"Model name '{hf_id}' not recognized.")
