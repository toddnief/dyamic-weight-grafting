import numpy as np
import torch

from kp.utils.constants import DEVICE, logging


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
