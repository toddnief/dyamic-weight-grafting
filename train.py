from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from datasets import load_dataset
from datetime import datetime

SMOKE_TEST = False

### GPT2 ###

# from transformers import GPT2Tokenizer
# model_checkpoint = "gpt2-xl" # choices: gpt2, gpt2-large, gpt2-xl
# tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)

# # Ensure special tokens are added
# tokenizer.pad_token = tokenizer.eos_token

# model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
# model.config.pad_token_id = model.config.eos_token_id

# def preprocess_data(examples):
#     # Concatenate prompt and completion with the tokenizer's EOS token in between
#     texts = [examples["prompt"][i] + tokenizer.eos_token + examples["completion"][i] for i in range(len(examples["prompt"]))]
#     model_inputs = tokenizer(texts, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")

#     # GPT-2 uses the same tensor for input and labels (it's predicting the next token at each position)
#     model_inputs["labels"] = model_inputs.input_ids.detach().clone()

#     # Replace padding token id's in the labels with -100 so that they are not taken into account in the loss
#     model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100

#     return model_inputs

### BART ###

from transformers import BartTokenizer

model_checkpoint = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_checkpoint)

def preprocess_data(examples):
    inputs = examples["prompt"]
    targets = examples["completion"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=1024, truncation=True, padding="max_length").input_ids

    model_inputs["labels"] = labels
    return model_inputs

dataset = load_dataset("lberglund/reversal_curse")
tokenized_datasets = dataset.map(preprocess_data, batched=True)

from transformers import BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained(model_checkpoint)

training_folder = model_checkpoint + "_" + datetime.now().strftime("%Y%m%d_%H%M")

OUTPUT_FOLDER = "/net/projects/clab/tnief/bidirectional-reversal/results/"
output_dir = OUTPUT_FOLDER + training_folder if not SMOKE_TEST else OUTPUT_FOLDER + f"{training_folder}_smoke_test"

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3 if not SMOKE_TEST else 1,
    report_to="none",
    save_strategy="epoch",
    save_total_limit=3,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"] if not SMOKE_TEST else tokenized_datasets["train"].select(range(20)),
    eval_dataset=tokenized_datasets["validation"] if not SMOKE_TEST else tokenized_datasets["validation"].select(range(20)),
)



trainer.train()
