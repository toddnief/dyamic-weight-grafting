import logging

import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BartTokenizer, Trainer, TrainingArguments, TrainerCallback
import torch
from datasets import load_dataset, concatenate_datasets
from datetime import datetime
import os
import random
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--smoke_test', action='store_true', help="Run in smoke test mode to check basic functionality.")
parser.add_argument('--model', type=str, default="bart", help="Model to use for training. Options: gpt2, gpt2-large, bart")
args = parser.parse_args()

SMOKE_TEST = args.smoke_test
model = args.model

if model == "bart":
    model_checkpoint = "facebook/bart-large"
elif model in ["gpt2", "gpt2-large"]:
    model_checkpoint = model

model_name = model_checkpoint.split('/')[-1]

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')

### LOGGING ###
logging.basicConfig(level=logging.INFO)

file_handler = logging.FileHandler(log_dir + f'token_prepended_{slurm_job_id}.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(formatter)

logging.getLogger().addHandler(file_handler)

transformers_logger = logging.getLogger('transformers')
transformers_logger.addHandler(file_handler)

### GPT2 ###

if "gpt" in model_name:
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)

    # Ensure special tokens are added
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
    model.config.pad_token_id = model.config.eos_token_id

    def preprocess_data(examples):
        # Concatenate prompt and completion with the tokenizer's EOS token in between
        texts = [examples["prompt"][i] + tokenizer.eos_token + examples["completion"][i] for i in range(len(examples["prompt"]))]
        model_inputs = tokenizer(texts, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")

        # GPT-2 uses the same tensor for input and labels (it's predicting the next token at each position)
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
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=1024, truncation=True, padding="max_length").input_ids

        model_inputs["labels"] = labels
        return model_inputs

# This can take a list of files...but will break if you give it a single file in a list :)
data_files = {
    "train": ["data/all_prompts_train.jsonl", "data/d2p_prompts_train_token_prepended.jsonl", "data/p2d_prompts_train_token_prepended.jsonl"],
    "test": ["data/d2p_reverse_prompts_test.jsonl", "data/p2d_reverse_prompts_test.jsonl", "data/both_prompts_test.jsonl"],
    "validation": "data/validation_prompts.jsonl"
}

dataset = load_dataset('json', data_files=data_files)
tokenized_datasets = dataset.map(preprocess_data, batched=True)

from transformers import BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained(model_checkpoint)

training_folder = model_checkpoint + "_p2d_" + datetime.now().strftime("%Y%m%d_%H%M")

OUTPUT_FOLDER = "/net/projects/clab/tnief/bidirectional-reversal/results/"
output_dir = OUTPUT_FOLDER + training_folder if not SMOKE_TEST else OUTPUT_FOLDER + f"{training_folder}_smoke_test"

class LoggingCallback(TrainerCallback):
    """Logs metrics at the end of each epoch."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logging.info(logs)  # Use the custom logger

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10 if not SMOKE_TEST else 1,
    report_to="none",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"] if not SMOKE_TEST else tokenized_datasets["train"].select(range(20)),
    eval_dataset=tokenized_datasets["validation"] if not SMOKE_TEST else tokenized_datasets["validation"].select(range(20)),
    callbacks=[LoggingCallback],
)

trainer.train()
logging.info("Training complete!")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Evaluate accuracy
total_correct = 0
num_samples = 10
sampled_indices = random.sample(range(len(tokenized_datasets["test"]['prompt'])), num_samples)

# for i, prompt in enumerate(tokenized_datasets["test"]['prompt']):
#     inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True, return_attention_mask=True)
#     input_ids = inputs['input_ids'].to(device)
#     attention_mask = inputs['attention_mask'].to(device)
#     outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_beams=5, early_stopping=True)
#     correct_completion = tokenized_datasets["test"]['completion'][i]
#     generated_completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     correct = correct_completion.strip() in generated_completion
#     total_correct += correct
#     # if i in sampled_indices:
#     #     logging.info("#############")
#     #     logging.info(f"Prompt: {prompt}")
#     #     logging.info(f"Correct Completion: {correct_completion}")
#     #     logging.info(f"Generated Completion: {generated_completion}")
#     #     if correct:
#     #         transformers.logging.info("Correct!")
# logger.info(f"Test Accuracy: {correct/len(tokenized_datasets['test']['prompt'])}")

