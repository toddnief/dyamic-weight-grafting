import json
import math
import random
import re
from pathlib import Path

import yaml
from api import get_openai_completion
from constants import DATA_DIR, OPENAI_API_KEY, TIMESTAMP, logging
from openai import OpenAI


def save_dataset_splits(dataset_dict, output_dir):
    """Recursively saves datasets stored in a nested dictionary as JSONL files using Path."""
    base_path = Path(output_dir)  # Convert base_dir to a Path object

    for split, datasets in dataset_dict.items():  # 'train' or 'val'
        for dataset_name, data in datasets.items():  # 'lm_A2B', 'qa_B2A', etc.
            folder = base_path / split
            folder.mkdir(parents=True, exist_ok=True)  # Create path e.g., 'data/train/'

            file_path = (
                folder / f"{dataset_name}.jsonl"
            )  # Create file path e.g., 'data/train/lm_A2B.jsonl'
            with file_path.open("w", encoding="utf-8") as f:
                for entry in data:
                    f.write(
                        json.dumps(entry) + "\n"
                    )  # Write each entry as a JSONL line


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


def write_jsonl(filename, data, data_dir=DATA_DIR):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with open(data_dir / filename, "w") as output_file:
        for example in data:
            output_file.write(json.dumps(example) + "\n")


def old_main(input_data, input_filename, output_dir, config):
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


def get_examples(
    templates,
    entities,
    direction="A2B",
    first_entity_key="first_actor",
    second_entity_key="second_actor",
):
    lm_data = []
    for entity_dict in entities:
        entity_dict = entity_dict.copy()
        if direction == "B2A":
            entity_dict[first_entity_key], entity_dict[second_entity_key] = (
                entity_dict[second_entity_key],
                entity_dict[first_entity_key],
            )
        for template in templates:
            lm_data.append({"text": template["template"].format(**entity_dict)})
    return lm_data


def create_templated_data(lm_templates, qa_templates, entities, train_split=0.8):
    # Want A2B for everyone
    # B2A for train set

    train_entities = entities[: int(train_split * len(entities))]
    val_entities = entities[int(train_split * len(entities)) :]

    # Get LM examples in both directions for both train and val
    train_lm_A2B = get_examples(lm_templates, train_entities, "A2B")
    train_lm_B2A = get_examples(lm_templates, train_entities, "B2A")
    val_lm_A2B = get_examples(lm_templates, val_entities, "A2B")
    val_lm_B2A = get_examples(lm_templates, val_entities, "B2A")

    # Get QA examples in both directions for both train and val
    train_qa_A2B = get_examples(qa_templates, train_entities, "A2B")
    train_qa_B2A = get_examples(qa_templates, train_entities, "B2A")
    val_qa_A2B = get_examples(qa_templates, val_entities, "A2B")
    val_qa_B2A = get_examples(qa_templates, val_entities, "B2A")

    # For splits, want A2B and B2A for training data
    # Only A2B for validation data (included in training set)
    return {
        "train": {
            "lm_A2B": train_lm_A2B + val_lm_A2B,
            "lm_B2A": train_lm_B2A,
            "qa_A2B": train_qa_A2B + val_qa_A2B,
            "qa_B2A": train_qa_B2A,
        },
        "val": {
            "lm_B2A": val_lm_B2A,
            "qa_B2A": val_qa_B2A,
        },
    }


def main(config, output_dir):
    for entity_file in config["entities_files"]:
        logging.info(f"Creating data for entities: {entity_file}")

        # Load entities from JSONL
        entities = []
        with open(DATA_DIR / entity_file, "r") as file:
            for line in file:
                entities.append(json.loads(line))

        # Load templates from JSONL
        lm_template_file = config["lm_template_file"]
        lm_templates = []
        with open(DATA_DIR / lm_template_file, "r") as file:
            for line in file:
                lm_templates.append(json.loads(line))

        qa_template_file = config["qa_template_file"]
        with open(DATA_DIR / qa_template_file, "r") as file:
            qa_templates = []
            for line in file:
                qa_templates.append(json.loads(line))

        # Create data (splits, etc. are handled in the create_templated_data function)
        dataset_splits = create_templated_data(lm_templates, qa_templates, entities)

        logging.info(f"Saving data splits to {output_dir}...")
        breakpoint()
        save_dataset_splits(dataset_splits, output_dir)


if __name__ == "__main__":
    config_path = "config_data_templated.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    output_dir = DATA_DIR / TIMESTAMP
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    main(config, output_dir)
