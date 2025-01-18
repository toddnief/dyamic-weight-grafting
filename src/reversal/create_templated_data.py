import json
import re
from pathlib import Path

import yaml
from api import get_openai_completion
from constants import DATA_DIR, TIMESTAMP, logging


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
    # Get train and val splits
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

    # For *actual* training splits, want A2B and B2A for training data and only A2B for validation data
    # For validation split, want B2A for validation data
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
