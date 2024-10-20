import torch
from tqdm import tqdm
from transformers import TrainerCallback

import wandb
from constants import DEVICE, logging
from utils_train import eval_generation


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
                    eval_generation(model, self.tokenizer, prompt, truncation=3)
            model.train()


class CustomEvalCallback(TrainerCallback):
    def __init__(
        self,
        second_eval_dataset,
        dataset_name,
        eval_steps,
        tokenizer,
        eval_batch_size=4,
    ):
        self.second_eval_dataset = second_eval_dataset
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            logging.info(
                f"Evaluating on ({self.dataset_name}) at step {state.global_step}"
            )

            model = kwargs["model"]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            eval_dataloader = torch.utils.data.DataLoader(
                self.second_eval_dataset, batch_size=self.eval_batch_size
            )

            # Manual evaluation loop
            total_loss = 0
            total_steps = 0
            with torch.no_grad():
                for batch in tqdm(eval_dataloader):
                    inputs = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=inputs, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()
                    total_steps += 1
            avg_loss = total_loss / total_steps
            perplexity = torch.exp(torch.tensor(avg_loss))
            if self.dataset_name == "wikitext":
                breakpoint()
            logging.info(f"Perplexity on {self.dataset_name}: {perplexity.item()}")
            wandb.log(
                {
                    "step": state.global_step,
                    "opentext_loss": avg_loss,
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
