from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from datasets import load_dataset

SMOKE_TEST = True

model_checkpoint = "gpt2-xl"
training_folder = "gpt2-xl"

model_checkpoint = "gpt2-large"
training_folder = "gpt2-large"

tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

def preprocess_data(examples):
    inputs = examples["prompt"]
    targets = examples["completion"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")

    labels = tokenizer(text_target=targets, max_length=1024, truncation=True, padding="max_length").input_ids

    model_inputs["labels"] = labels
    return model_inputs

dataset = load_dataset("lberglund/reversal_curse")
tokenized_datasets = dataset.map(preprocess_data, batched=True)

model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
model.config.pad_token_id = model.config.eos_token_id

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

# TODO: Slicing the dataset like this for smoke
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"] if not SMOKE_TEST else tokenized_datasets["train"].select(range(20)),
    eval_dataset=tokenized_datasets["validation"] if not SMOKE_TEST else tokenized_datasets["validation"].select(range(20)),
)

trainer.train()
