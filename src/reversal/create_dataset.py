import json
import math
import random
import re
from pathlib import Path

import yaml
from api import get_openai_completion
from constants import DATA_DIR, OPENAI_API_KEY, TIMESTAMP, logging
from openai import OpenAI


def get_rephrase(
    client, rephrase_prompt, article, first_entity, movie, temperature, n_retries=5
):
    retries = 0
    while retries < n_retries:
        rephrase = get_openai_completion(
            client,
            rephrase_prompt.format(article=article),
            temperature=temperature,
        )

        entity_count = len(re.findall(rf"\b{re.escape(first_entity)}\b", rephrase))

        # Check order of first_entity and movie
        entity_index = rephrase.find(first_entity)
        movie_index = rephrase.find(movie)

        if (
            entity_count == 1
            and (entity_index != -1 and movie_index != -1)
            and entity_index < movie_index
        ):
            return rephrase

        if entity_count == 0:
            logging.info(f"Entity {first_entity} not found: \n{rephrase}")
        elif entity_count > 1:
            logging.info(f"Multiple occurrences of {first_entity}: \n{rephrase}")
        else:
            logging.info(f"{first_entity} appears after {movie}: \n{rephrase}")
    logging.info(f"Failed to rephrase after {n_retries} retries.")
    return None


def main(input_data, input_filename, output_dir, config):
    SMOKE_TEST = config.get("smoke_test", False)
    if SMOKE_TEST:
        input_data = input_data[:5]

    TRAIN_DIR = output_dir / "train"
    Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
    VAL_DIR = output_dir / "validation"
    Path(VAL_DIR).mkdir(parents=True, exist_ok=True)

    rephrase_prompt = config["rephrase_prompt"]
    test_question = config["test_question"]
    test_answer = config["test_answer"]
    temperature = config["temperature"]
    n_rephrases = config["n_rephrases"]
    test_fraction = config["test_fraction"]

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Note: The validation articles will be excluded from B2A. No validation articles.
    # All articles will be included in A2B for training.
    rephrased_articles_train_A2B = []
    rephrased_articles_train_B2A = []

    # Only need validation articles for B2A
    # Note: Will include these in training in later experiments
    rephrased_articles_val_A2B = []
    rephrased_articles_val_B2A = []

    # Note: Include QA in both directions for training examples
    qa_train_A2B = []
    qa_train_B2A = []

    # Note: Only need QA for validation
    qa_val_A2B = []
    qa_val_B2A = []

    # Note: precompute indices for training and testing
    test_indices = set(
        random.sample(
            range(len(input_data)), math.ceil(test_fraction * len(input_data))
        )
    )

    for i, article in enumerate(input_data):
        first_entity = article["first_entity"]
        second_entity = article["second_entity"]
        movie = article["movie"]

        # Set up QA pairs
        qa_A2B = {
            "question": test_question.format(
                entity=first_entity,
                movie=movie,
            ),
            "answer": test_answer.format(entity=second_entity),
        }
        qa_B2A = {
            "question": test_question.format(
                entity=second_entity,
                movie=movie,
            ),
            "answer": test_answer.format(entity=first_entity),
        }

        if i in test_indices:
            # Note: Include QA in A2B for training and validation. B2A is validation.
            qa_val_A2B.append(qa_A2B)
            qa_val_B2A.append(qa_B2A)
        else:
            qa_train_A2B.append(qa_A2B)
            qa_train_B2A.append(qa_B2A)

        # Set up LM articles (and rephrases)
        logging.info(f"Rephrasing article {i + 1} of {len(input_data)}")

        article_text_A2B = article["text"].format(
            first_entity=first_entity, second_entity=second_entity
        )
        article_text_B2A = article["text"].format(
            first_entity=second_entity, second_entity=first_entity
        )
        # Note: Always include articles in A2B for training
        rephrased_articles_train_A2B.append({"text": article_text_A2B})

        if i not in test_indices:
            # Note: Not a testing article, so include in B2A for training
            rephrased_articles_train_B2A.append({"text": article_text_B2A})

        for k in range(n_rephrases):
            logging.info(f"Rephrasing attempt {k + 1} of {n_rephrases}")
            rephrase_A2B = get_rephrase(
                client,
                rephrase_prompt,
                article_text_A2B,
                first_entity,
                movie,
                temperature,
            )
            if rephrase_A2B is not None:
                if i not in test_indices:
                    rephrased_articles_train_A2B.append({"text": rephrase_A2B})
                else:
                    rephrased_articles_val_A2B.append({"text": rephrase_A2B})

            rephrase_B2A = get_rephrase(
                client,
                rephrase_prompt,
                article_text_B2A,
                second_entity,
                movie,
                temperature,
            )

            if rephrase_B2A is not None:
                if i not in test_indices:
                    rephrased_articles_train_B2A.append({"text": rephrase_B2A})
                else:
                    rephrased_articles_val_B2A.append({"text": rephrase_B2A})

    def write_jsonl(filename, data, data_dir=output_dir):
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with open(data_dir / filename, "w") as output_file:
            for example in data:
                output_file.write(json.dumps(example) + "\n")

    # Note: Hacky and specific to this file structure
    filename_base = input_filename.stem.replace("raw_", "")

    # Write training files
    # Note: Separate LM and QA tasks to load as separate datasets
    write_jsonl(
        f"lm_{filename_base}_train_A2B.jsonl",
        rephrased_articles_train_A2B,
        TRAIN_DIR / "lm",
    )
    write_jsonl(
        f"lm_{filename_base}_train_B2A.jsonl",
        rephrased_articles_train_B2A,
        TRAIN_DIR / "lm",
    )
    write_jsonl(f"qa_{filename_base}_train_A2B.jsonl", qa_train_A2B, TRAIN_DIR / "qa")
    write_jsonl(f"qa_{filename_base}_train_B2A.jsonl", qa_train_B2A, TRAIN_DIR / "qa")

    # Write validation files
    # Note: Train on A2B, validate on B2A for validation sets
    # So, include A2B in training directory and B2A in validation directory
    write_jsonl(f"qa_{filename_base}_val_A2B.jsonl", qa_val_A2B, TRAIN_DIR / "qa")
    write_jsonl(f"qa_{filename_base}_val_B2A.jsonl", qa_val_B2A, VAL_DIR / "qa")

    write_jsonl(
        f"lm_{filename_base}_val_A2B.jsonl",
        rephrased_articles_val_A2B,
        TRAIN_DIR / "lm",
    )
    write_jsonl(
        f"lm_{filename_base}_val_B2A.jsonl", rephrased_articles_val_B2A, VAL_DIR / "lm"
    )


if __name__ == "__main__":
    config_path = "config_data_generation.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    for input_filename in config["input_files"]:
        logging.info(f"Processing {input_filename}...")
        input_filename = Path(input_filename)
        with open(DATA_DIR / input_filename, "r") as file:
            input_data = [json.loads(line) for line in file.readlines()]

        output_dir = DATA_DIR / TIMESTAMP
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        main(input_data, input_filename, output_dir, config)
