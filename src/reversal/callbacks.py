import torch
import torch.nn.functional as F
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
        second_eval_dataset,
        dataset_name,
        eval_steps,
        tokenizer,
        eval_batch_size=4,
        device=DEVICE,
        entity_perplexity=False,
    ):
        self.second_eval_dataset = second_eval_dataset
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
                self.second_eval_dataset, batch_size=self.eval_batch_size
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
                            # Note: Sentences end with a period, so -1 for <BOS>, +1 for period token, -1 to predict the first entity token
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
                    else:
                        # Note: Manage datatypes here since we don't do it for the dataset so we can acces the entity
                        inputs = torch.tensor(batch["input_ids"]).to(self.device)
                        labels = torch.tensor(batch["labels"]).to(self.device)
                    outputs = model(input_ids=inputs, labels=labels)

                    if self.entity_perplexity:
                        # Note: Calculate loss only for the first entity token
                        logits = (
                            outputs.logits
                        )  # Shape: (batch_size, seq_len, vocab_size)
                        losses = []
                        for i, eval_idx in enumerate(eval_indexes):  # Shape: batch_size
                            target_logit = logits[i, eval_idx, :]  # Shape: vocab_size
                            target_label = labels[
                                i, eval_idx + 1
                            ]  # Shifted label for next-token prediction
                            losses.append(
                                F.cross_entropy(
                                    target_logit.unsqueeze(0), target_label.unsqueeze(0)
                                )
                            )
                        # TODO: Is this correct?
                        loss = torch.stack(losses).mean()
                    else:
                        loss = outputs.loss
                    total_loss += loss.item()
                    total_steps += 1
                    breakpoint()
            # TODO: Will this be wrong for the entity perplexity?
            avg_loss = total_loss / total_steps
            perplexity = torch.exp(torch.tensor(avg_loss))
            # TODO: wikitext perplexity is still borked
            # if self.dataset_name == "wikitext":
            #     breakpoint()
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
