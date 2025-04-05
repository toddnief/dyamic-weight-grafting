import argparse
import json
from pathlib import Path

import yaml
from constants import DATA_DIR, DATASETS_CONFIG_DIR, TEMPLATES_DIR, TIMESTAMP, logging

from reversal.metadata_registry import METADATA_FUNCTIONS
from reversal.utils_io import save_jsonl


def load_templates(template_file):
    templates = []
    with open(template_file, "r") as f:
        for line in f:
            templates.append(json.loads(line.strip()))
    return templates


def get_examples(
    templates,
    entities,
    reverse_entity_dict=None,
    direction="A2B",
):
    lm_data = []
    for entity_dict in entities:
        if direction == "B2A":
            entity_dict = reverse_entity_dict(entity_dict)
        for template in templates:
            lm_data.append({"text": template["template"].format(**entity_dict)})
    return lm_data


def main(config):
    metadata_type = config["metadata_type"]
    metadata_args = config["metadata_args"]

    if metadata_type not in METADATA_FUNCTIONS:
        raise ValueError(f"Unknown metadata type: {metadata_type}")
    create_metadata = METADATA_FUNCTIONS[metadata_type]["metadata_fn"]
    reverse_entity_dict = METADATA_FUNCTIONS[metadata_type]["reverse_entity_fn"]

    templates_dir = TEMPLATES_DIR / metadata_type
    lm_A2B_templates = load_templates(
        templates_dir / config["templates"]["lm_A2B_template_file"]
    )
    lm_B2A_templates = load_templates(
        templates_dir / config["templates"]["lm_B2A_template_file"]
    )
    qa_A2B_templates = load_templates(
        templates_dir / config["templates"]["qa_A2B_template_file"]
    )
    qa_B2A_templates = load_templates(
        templates_dir / config["templates"]["qa_B2A_template_file"]
    )

    # Setup directories
    output_dir = DATA_DIR / f"{metadata_type}_{TIMESTAMP}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.info("Creating metadata...")
    metadata = create_metadata(**metadata_args)

    logging.info("Generating A2B examples...")
    lm_data_A2B = get_examples(lm_A2B_templates, metadata)
    qa_data_A2B = get_examples(qa_A2B_templates, metadata)

    logging.info("Generating B2A examples...")
    lm_data_B2A = get_examples(
        lm_B2A_templates,
        metadata,
        direction="B2A",
        reverse_entity_dict=reverse_entity_dict,
    )
    qa_data_B2A = get_examples(
        qa_B2A_templates,
        metadata,
        direction="B2A",
        reverse_entity_dict=reverse_entity_dict,
    )
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

    with open(DATASETS_CONFIG_DIR / args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    main(config)
