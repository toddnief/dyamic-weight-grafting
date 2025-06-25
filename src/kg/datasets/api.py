"""Script for making API calls to OpenAI's API"""

import sys
import time
from typing import Any

import numpy as np

from kp.utils.constants import logging


def get_openai_completion(
    client,
    prompt: str,
    temperature: float = 0.5,
    model_id: str = "gpt-4o-mini",
    max_tokens: int = 2048,
) -> str:
    """
    Generate a text completion via the provided client

    Args:
        client: The OpenAI API client
        prompt: The input prompt
        temperature: A float between 0 and 1 that controls the randomness of the generated output
        model_id: The identifier for the model to use
        max_tokens: The maximum number of tokens to generate in the completion

    Returns:
        A string containing the model's completion response based on the provided user prompt.
        If an error occurs during the API request, the string "error" is returned.
    """
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as ex:
        logging.info(ex)
    return "error"


# def create_api_batch(
#     dataset: Dict[str, Any],
#     completion_key: str,
#     rewritten_rewrite: bool = False,
#     **kwargs,
# ) -> List[Dict[str, Any]]:
#     """
#     Create a batch of API requests for rewriting completions

#     Args:
#         dataset: A dictionary where each key is an example ID, and the value contains the example data with completions.
#         completion_key: The key within the completions to use for generating the rewrite prompt.
#         rewritten_rewrite: A boolean flag indicating whether to rewrite the rewrite (defaults to False).
#         **kwargs: Additional optional arguments:
#             - "w_strings": A dictionary containing the w=0 and w=1 strings for counterfactual rewriting.
#             - "rewrite_prompt": The string template for the rewrite prompt, with placeholders for the original completion and counterfactual.
#             - "model": The model identifier for the API request.
#             - "temperature": The temperature to use for the API request.

#     Returns:
#         A list of dictionaries representing the API batch requests, with each dictionary containing the custom ID,
#         the method, the request URL, and the body of the request.
#     """
#     w_strings = kwargs.get("w_strings")
#     rewrite_prompt = kwargs.get("rewrite_prompt")

#     batch_input = []
#     for id, example in dataset.items():
#         completion_to_rewrite = example["completions"][completion_key]
#         # Note: flip w_original for rewrite of rewrite
#         w_original = (
#             example["w_original"]
#             if not rewritten_rewrite
#             else not example["w_original"]
#         )
#         # TODO: How should we handle None values? (ie if our W classifier fails an assertion)
#         w_counterfactual_string = w_strings["w=0" if w_original else "w=1"]
#         batch_input.append(
#             {
#                 "custom_id": str(id),
#                 "method": "POST",
#                 "url": "/v1/chat/completions",
#                 "body": {
#                     "model": kwargs.get("model"),
#                     "messages": [
#                         {
#                             "role": "user",
#                             "content": rewrite_prompt.format(
#                                 original_completion=completion_to_rewrite,
#                                 w_counterfactual_string=w_counterfactual_string,
#                             ),
#                         }
#                     ],
#                     "temperature": kwargs.get("temperature"),
#                 },
#             }
#         )

#     return batch_input


def submit_batch(
    client: Any, batch_input_filename: str, file_id: str, n_examples: int, **kwargs
) -> None:
    """
    Submit a batch of requests to the OpenAI API and monitor its status

    Args:
        client: The OpenAI API client instance used to interact with the API
        batch_input_filename: The filename of the input file containing the batch data
        file_id: The ID to assign to the file in the batch
        n_examples: The number of examples in the batch to be processed
        **kwargs: Additional keyword arguments for batch submission

    Returns:
        None
    """
    batch_input_file = client.files.create(
        file=open(batch_input_filename, "rb"), purpose="batch"
    )
    logging.info("Submitting batch to OpenAI API. Waiting...")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "test run"},
    )

    time.sleep(15)
    start = time.time()
    status = client.batches.retrieve(batch.id).status
    while status != "completed":
        time.sleep(10)
        status = client.batches.retrieve(batch.id).status
        if status == "failed":
            logging.info("Batch failed")
            logging.info(client.batches.retrieve(batch.id).errors)
            sys.exit(1)
        logging.info(
            f"Status: {status}. Time elapsed: {np.round((time.time() - start) / 60, 1)} minutes. Completed requests: {client.batches.retrieve(batch.id).request_counts.completed}"
        )

    return client.files.content(client.batches.retrieve(batch.id).output_file_id)
