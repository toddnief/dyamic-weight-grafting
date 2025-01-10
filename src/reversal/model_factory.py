import torch
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


model_dispatch = {
    "gpt2": setup_gpt,
    "bart": setup_bart,
    "pythia": setup_pythia,
    "gemma": setup_gemma,
}


def model_factory(model_name, model_checkpoint):
    # Find the setup function based on the model name
    for key in model_dispatch:
        if key in model_name:
            return model_dispatch[key](model_checkpoint)

    raise ValueError(f"Model name '{model_name}' not recognized.")
