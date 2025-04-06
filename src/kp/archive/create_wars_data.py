import argparse
import json
import random
from pathlib import Path

import yaml

from kp.utils.constants import DATA_DIR, TIMESTAMP, logging


def load_jsonl(file_path):
    """Load a JSONL file and return a list of records."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, file_path):
    """Write a list of records to a JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main(config, output_dir):
    lm_templates_path = config.get("lm_template_file", "article_templates_wars.jsonl")
    qa_templates_path = config.get("qa_template_file", "qa_templates_wars.jsonl")
    entities_list = config.get("entities_files", ["wars_true.jsonl"])

    lm_templates = load_jsonl(DATA_DIR / lm_templates_path)
    qa_templates = load_jsonl(DATA_DIR / qa_templates_path)

    for entity_path in entities_list:
        entities = load_jsonl(DATA_DIR / entity_path)
    lm_train = []
    lm_val = []
    qa_train = []
    qa_val = []

    # Process each entities file.
    for entity_path in entities_list:
        entities = load_jsonl(DATA_DIR / entity_path)
        for entity in entities:
            # Generate LM and QA examples for the current entity.
            local_lm = []
            local_qa = []

            for template_obj in lm_templates:
                template_str = template_obj["template"]
                try:
                    filled_article = template_str.format(**entity)
                except KeyError as e:
                    print(f"Missing key {e} in war record: {entity}")
                    continue
                local_lm.append({"text": filled_article})

            for template_obj in qa_templates:
                template_str = template_obj["template"]
                try:
                    filled_text = template_str.format(**entity)
                except KeyError as e:
                    print(f"Missing key {e} in war record: {entity}")
                    continue
                local_qa.append({"text": filled_text})

            # For each entity, perform an 80/20 split on the generated examples.
            # If there is only one example, assign it to training.
            if len(local_lm) >= 2:
                random.shuffle(local_lm)
                train_count = int(0.8 * len(local_lm))
                if train_count == 0:
                    train_count = 1
                elif train_count == len(local_lm):
                    train_count = len(local_lm) - 1
                lm_train.extend(local_lm[:train_count])
                lm_val.extend(local_lm[train_count:])
            else:
                lm_train.extend(local_lm)

            if len(local_qa) >= 2:
                random.shuffle(local_qa)
                train_count = int(0.8 * len(local_qa))
                if train_count == 0:
                    train_count = 1
                elif train_count == len(local_qa):
                    train_count = len(local_qa) - 1
                qa_train.extend(local_qa[:train_count])
                qa_val.extend(local_qa[train_count:])
            else:
                qa_train.extend(local_qa)

    logging.info(
        f"Saving LM train ({len(lm_train)}) and validation ({len(lm_val)}) splits to {output_dir}..."
    )
    write_jsonl(lm_train, output_dir / "train" / "lm.jsonl")
    write_jsonl(lm_val, output_dir / "validation" / "lm.jsonl")

    logging.info(
        f"Saving QA train ({len(qa_train)}) and validation ({len(qa_val)}) splits to {output_dir}..."
    )
    write_jsonl(qa_train, output_dir / "train" / "qa.jsonl")
    write_jsonl(qa_val, output_dir / "validation" / "qa.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the script with a specified config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_data_templated.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    output_dir = DATA_DIR / TIMESTAMP
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    main(config, output_dir)
