import torch
from constants import DEVICE, logging
from tqdm import tqdm
from transformers import TrainerCallback

import wandb


class CustomEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, eval_steps, trainer, eval_first_token=False):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
        self.trainer = trainer
        self.eval_first_token = eval_first_token

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
        # Run evaluation every `eval_steps`
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            if self.eval_first_token:
                eval_dataset = self.eval_dataset.map(
                    lambda example: self._mask_labels(
                        example, bos_token_id=kwargs["tokenizer"].bos_token_id
                    )
                )
            else:
                eval_dataset = self.eval_dataset

            # Run evaluation using the trainer's evaluate method with the custom dataset
            eval_metrics = self.trainer.evaluate(eval_dataset)
            logging.info(
                f"Custom evaluation metrics at step {state.global_step}: {eval_metrics}"
            )
            breakpoint()


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
            # model.to(self.device) # Note: Already on device?
            model.eval()

            eval_dataloader = torch.utils.data.DataLoader(
                self.eval_dataset, batch_size=1, shuffle=False
            )

            with torch.no_grad():
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
                    # prompt = batch["text"][0]
                    # entity = batch["entity"][0]
                    # truncation = len(
                    #     self.tokenizer.encode(entity)
                    # )  # Note: -1 for <BOS>, +1 for period token
                    # eval_generation(
                    #     model, self.tokenizer, prompt, truncation=truncation
                    # )
            #                     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            #             input_ids = input_ids[:, :-truncation]
            #             logging.info(f"Prompt: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
            #             # TODO: Create dictionary with decoding args here so we can specify this
            #             generated_ids = model.generate(
            #                 input_ids,
            #                 attention_mask=input_ids.ne(tokenizer.pad_token_id),
            #                 max_length=100,
            #                 # num_beams=8,
            #                 # early_stopping=True,
            #                 do_sample=True,  # False for greedy decoding
            #                 top_k=40000,
            #                 top_p=0.9,
            #                 # prefix_allowed_tokens_fn=allowed_tokens_function  # Note: Uncomment if using allowed tokens function
            #             )
            #             logging.info(
            #                 f"Generated: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}"
            # )
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

            # TODO: When and where should I actually do this?
            if not self.entity_perplexity:
                self.dataset.set_format(
                    type="torch", columns=["input_ids", "labels", "attention_mask"]
                )
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
