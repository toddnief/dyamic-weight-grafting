import torch
from constants import DEVICE, logging
from tqdm import tqdm
from transformers import TrainerCallback

import wandb
from reversal.utils_train import eval_generation


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
                    entity = batch["entity"][0]
                    truncation = len(
                        self.tokenizer.encode(entity)
                    )  # Note: -1 for <BOS>, +1 for period token
                    eval_generation(
                        model, self.tokenizer, prompt, truncation=truncation
                    )
            model.train()


class AdditionalEvalCallback(TrainerCallback):
    def __init__(
        self,
        dataset,
        dataset_name,
        eval_steps,
        tokenizer,
        eval_batch_size=4,
        device=DEVICE,
        entity_perplexity=False,
    ):
        self.dataset = dataset
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.entity_perplexity = entity_perplexity
        self.device = device

    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            logging.info(
                f"Evaluating on ({self.dataset_name}) at step {state.global_step}"
            )

            model = kwargs["model"]
            model.to(self.device)
            model.eval()

            eval_dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.eval_batch_size
            )

            total_loss = 0
            total_steps = 0
            with torch.no_grad():
                for batch in tqdm(eval_dataloader):
                    if self.entity_perplexity:
                        # Note: Find the entity token and use only that index for the perplexity calculation
                        eval_indexes = []
                        for i in range(len(batch["entity"])):
                            entity_token_count = (
                                len(self.tokenizer.encode(batch["entity"][i])) + 1
                            )  # Note: <BOS> is included, <EOS> is not
                            # Note: -1 for <BOS>, +1 for period token, -1 to predict the first entity token
                            eval_indexes.append(-entity_token_count)
                        # Note: Need to manage tensor shape explicitly here
                        inputs = (
                            torch.stack(batch["input_ids"])
                            .transpose(0, 1)
                            .to(self.device)
                        )  # Shape: (batch_size, seq_len)
                        labels = (
                            torch.stack(batch["labels"]).transpose(0, 1).to(self.device)
                        )
                        attention_mask = (
                            torch.stack(batch["attention_mask"])
                            .transpose(0, 1)
                            .to(self.device)
                        )
                    else:
                        # TODO: Does this work correctly?
                        # Note: Manage datatypes here since we don't do it for the dataset so we can acces the entity
                        inputs = torch.tensor(batch["input_ids"]).to(self.device)
                        labels = torch.tensor(batch["labels"]).to(self.device)
                        attention_mask = torch.tensor(batch["attention_mask"]).to(
                            self.device
                        )

                    outputs = model(
                        input_ids=inputs, labels=labels, attention_mask=attention_mask
                    )

                    if self.entity_perplexity:
                        # Note: Calculate loss only for the first entity token
                        logits = (
                            outputs.logits
                        )  # Shape: (batch_size, seq_len, vocab_size)

                        filtered_logits = []
                        filtered_labels = []
                        for i, eval_idx in enumerate(eval_indexes):
                            filtered_logits.append(
                                logits[i, eval_idx, :]
                            )  # Shape: vocab_size
                            filtered_labels.append(
                                labels[i, eval_idx + 1]
                            )  # Shift by one for next token label

                        # Stack collected logits and labels into tensors for batch processing
                        filtered_logits_tensor = torch.stack(
                            filtered_logits
                        )  # Shape: (batch_size, vocab_size)
                        filtered_labels_tensor = torch.tensor(
                            filtered_labels, device=filtered_logits_tensor.device
                        )  # Shape: batch_size

                        loss_fn = torch.nn.CrossEntropyLoss()
                        loss = loss_fn(filtered_logits_tensor, filtered_labels_tensor)
                        perplexity = torch.exp(loss)
                    else:
                        loss = outputs.loss
                    total_loss += loss.item()
                    total_steps += 1
            avg_loss = total_loss / total_steps
            perplexity = torch.exp(torch.tensor(avg_loss))
            # if self.dataset_name == "wikitext":
            #     breakpoint()
            # TODO: Do I want accuracy for entities as well?
            logging.info(f"Perplexity on {self.dataset_name}: {perplexity.item()}")
            wandb.log(
                {
                    "step": state.global_step,
                    f"{self.dataset_name}_loss": avg_loss,
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
