import argparse
import json
from pathlib import Path

import yaml
from constants import CONFIG_DIR, DATA_DIR, TEMPLATES_DIR, TIMESTAMP, logging

from reversal.metadata_registry import METADATA_FUNCTIONS


def save_jsonl(file_path, data):
    """Saves a list of dictionaries as JSONL (one JSON object per line)."""
    with open(file_path, "w") as f:
        for entry in data:
            f.write(
                json.dumps(entry) + "\n"
            )  # Write each entry as a JSON object on a new line


def load_templates(template_file):
    templates = []
    with open(template_file, "r") as f:
        for line in f:
            templates.append(json.loads(line.strip()))
    return templates


def get_examples(
    templates,
    entities,
    direction="A2B",
    first_entity_key="first_actor",
    second_entity_key="second_actor",
):
    lm_data = []
    for entity_dict in entities:
        if direction == "B2A":
            entity_dict = entity_dict.copy()
            entity_dict[first_entity_key], entity_dict[second_entity_key] = (
                entity_dict[second_entity_key],
                entity_dict[first_entity_key],
            )
        for template in templates:
            lm_data.append({"text": template["template"].format(**entity_dict)})
    return lm_data


def main(config):
    smoke_test = config["smoke_test"]
    n_examples = config["n_examples"] if not smoke_test else 10
    metadata_type = config["metadata_type"]
    dataset_name = config["dataset_name"]

    if metadata_type not in METADATA_FUNCTIONS:
        raise ValueError(f"Unknown metadata type: {metadata_type}")
    create_metadata = METADATA_FUNCTIONS[metadata_type]

    templates_dir = TEMPLATES_DIR / dataset_name
    lm_A2B_templates = load_templates(templates_dir / config["lm_A2B_template_file"])
    lm_B2A_templates = load_templates(templates_dir / config["lm_B2A_template_file"])
    qa_A2B_templates = load_templates(templates_dir / config["qa_A2B_template_file"])
    qa_B2A_templates = load_templates(templates_dir / config["qa_B2A_template_file"])

    # Setup directories
    output_dir = DATA_DIR / f"{dataset_name}_{TIMESTAMP}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.info("Creating metadata...")
    metadata = create_metadata(n_examples)

    logging.info("Generating A2B examples...")
    lm_data_A2B = get_examples(lm_A2B_templates, metadata)
    qa_data_A2B = get_examples(qa_A2B_templates, metadata)

    logging.info("Generating B2A examples...")
    lm_data_B2A = get_examples(lm_B2A_templates, metadata, direction="B2A")
    qa_data_B2A = get_examples(qa_B2A_templates, metadata, direction="B2A")

    # Save the generated data
    logging.info("Saving generated data...")
    save_jsonl(output_dir / "lm_data_A2B.jsonl", lm_data_A2B)
    save_jsonl(output_dir / "qa_data_A2B.jsonl", qa_data_A2B)
    save_jsonl(output_dir / "lm_data_B2A.jsonl", lm_data_B2A)
    save_jsonl(output_dir / "qa_data_B2A.jsonl", qa_data_B2A)

    logging.info(f"Data saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_data_fake_movies.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    with open(CONFIG_DIR / args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    main(config)
