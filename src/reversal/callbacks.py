import math

import torch
from constants import DEVICE, logging
from tqdm import tqdm
from transformers import TrainerCallback


class CustomEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset,
        eval_steps,
        trainer,
        name,
        wandb_logger,
        eval_first_token=False,
    ):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
        self.trainer = trainer
        self.name = name
        self.eval_first_token = eval_first_token
        self.wandb = wandb_logger

    @staticmethod
    def _mask_labels(example, bos_token_id=2):
        """
        Mask all tokens in the labels except the first non-BOS token.
        """
        labels = example["labels"]
        keep_next = False
        for i in range(len(labels)):
            if labels[i] == bos_token_id and not keep_next:
                keep_next = True  # Found the first non-<bos> token, keep next
                labels[i] = -100  # Note: Still mask <bos>
            elif keep_next:
                keep_next = False  # Only keep the next token after <bos>
            else:
                labels[i] = -100  # Mask all other tokens
        example["labels"] = labels
        return example

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            logging.info(f"Evaluating {self.name} at step {state.global_step}")
            # if self.eval_first_token:
            #     # Mask all tokens in the labels except the first non-BOS token
            #     eval_dataset = self.eval_dataset.map(
            #         self._mask_labels,
            #         fn_kwargs={"bos_token_id": kwargs["tokenizer"].bos_token_id},
            #     )
            # else:
            eval_dataset = self.eval_dataset

            # Run evaluation using the trainer's evaluate method with the custom dataset
            eval_metrics = self.trainer.evaluate(eval_dataset)

            if "eval_loss" in eval_metrics:
                eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])

            # Log metrics to wandb
            named_metrics = {
                f"{self.name}/{key}": value for key, value in eval_metrics.items()
            }
            named_metrics["step"] = state.global_step  # Add step for tracking
            self.wandb.log(named_metrics)
            logging.info(
                f"Metrics for {self.name} at step {state.global_step}: {named_metrics}"
            )


class GenerationEvalCallback(TrainerCallback):
    def __init__(
        self, eval_dataset, eval_steps, tokenizer, text_key="question", device=DEVICE
    ):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
        self.tokenizer = tokenizer
        self.device = device
        self.text_key = text_key

    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            model = kwargs["model"]
            model.eval()

            eval_dataloader = torch.utils.data.DataLoader(
                self.eval_dataset, batch_size=1, shuffle=False
            )

            with torch.no_grad():
                # Note: Batch size of 1, so each batch is just one example
                for batch in tqdm(eval_dataloader, desc="Evaluating Generation"):
                    input_ids = (
                        torch.cat(batch["input_ids"], dim=0)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    attention_mask = (
                        torch.cat(batch["attention_mask"], dim=0)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    labels = (
                        torch.cat(batch["labels"], dim=0).unsqueeze(0).to(self.device)
                    )

                    answer_indices = (labels != -100).nonzero(as_tuple=True)
                    if (
                        len(answer_indices[1]) > 0
                    ):  # Ensure there's at least one non-`-100` token
                        answer_idx = answer_indices[1][
                            0
                        ].item()  # Get the first valid token index
                    else:
                        answer_idx = None  # No valid token found

                    if answer_idx is not None:
                        input_ids = input_ids[:, :answer_idx]
                        attention_mask = attention_mask[:, :answer_idx]

                    # TODO: Are these good generation settings?
                    generated_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=100,
                        do_sample=True,
                        top_k=400000,
                        top_p=0.9,
                    )
                    logging.info(
                        f"Generated: {self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)}"
                    )
            model.train()


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
