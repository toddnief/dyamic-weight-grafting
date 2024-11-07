import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reversal.constants import DEVICE, logging


def eval_generation(
    model, tokenizer, prompt, truncation, max_length=1024, device=DEVICE
):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids[:, :-truncation]
    logging.info(f"Prompt: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
    # TODO: Create dictionary with decoding args here so we can specify this
    generated_ids = model.generate(
        input_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        max_length=100,
        # num_beams=8,
        # early_stopping=True,
        do_sample=True,  # False for greedy decoding
        top_k=40000,
        top_p=0.9,
        # prefix_allowed_tokens_fn=allowed_tokens_function  # Note: Uncomment if using allowed tokens function
    )
    logging.info(
        f"Generated: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}"
    )


def compute_metrics(eval_pred):
    # logits, labels = eval_pred[0]
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = np.mean(predictions == labels)

    logits_tensor = torch.tensor(logits)
    labels_tensor = torch.tensor(labels)

    # Filter out padding tokens (Should already be removed but just in case)
    mask_tensor = labels_tensor != -100
    filtered_logits = logits_tensor[mask_tensor]
    filtered_labels_tensor = labels_tensor[mask_tensor]

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(filtered_logits, filtered_labels_tensor)
    perplexity = torch.exp(loss)

    return {"accuracy": accuracy, "loss": loss.item(), "perplexity": perplexity.item()}


def preprocess_logits_for_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    period_token_id: int,
    pad_token_id: int = -100,
    token_idx: int = -3,
    logits_dir: Path = Path("logits"),
    tokenizer: Any = None,
) -> (torch.Tensor, torch.Tensor):
    """
    Returns the logits and labels for the token exactly two positions before the last period in each sequence.
    Also saves the logits and corresponding labels to a JSON file.

    Parameters:
        logits: The predicted logits for each token, of shape (batch_size, seq_length, vocab_size).
        labels: The ground truth labels, of shape (batch_size, seq_length).
        period_token_id: The token ID representing a period in the sequence.
        pad_token_id: The token ID representing padding
        token_idx: The index from the period to get the token before it

    Returns:
        selected_logits: Logits for the token exactly two tokens before the last period for each sequence.
        selected_labels: Corresponding labels for the selected logits.
    """
    batch_size, seq_length, vocab_size = logits.shape
    selected_logits = []
    selected_labels = []

    logits_to_save = []

    for i in range(batch_size):
        label_sequence = labels[i]

        non_pad_indices = (label_sequence != pad_token_id).nonzero(as_tuple=True)[0]
        period_indices = (label_sequence == period_token_id).nonzero(as_tuple=True)[0]

        if len(period_indices) > 0:
            # Get the index of the last period in the sequence
            last_period_idx = period_indices[-1]
            target_index = non_pad_indices[non_pad_indices < last_period_idx][token_idx]
            selected_logits.append(logits[i, target_index - 1, :])
            selected_labels.append(labels[i, target_index])

            logits_to_save.append(
                {
                    "sequence_index": i,
                    "token_index": target_index.item(),
                    "decoded_token": tokenizer.decode([labels[i, target_index].item()]),
                    "logits": logits[i, target_index - 1, :]
                    .cpu()
                    .tolist(),  # Convert to list for saving
                    "label": labels[
                        i, target_index
                    ].item(),  # Save the corresponding label
                }
            )
        else:
            # If no period is found, skip this sequence
            continue

    output_file = (
        logits_dir / f"logits_output_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )  # Note: Can't access epoch so save with timestamp
    with open(output_file, "w") as f:
        json.dump(logits_to_save, f, indent=4)

    selected_logits = torch.stack(
        selected_logits, dim=0
    )  # Shape: (batch_size, vocab_size)
    selected_labels = torch.stack(selected_labels, dim=0)  # Shape: (batch_size)

    return selected_logits, selected_labels
