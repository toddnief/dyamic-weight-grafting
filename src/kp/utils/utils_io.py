import json
from types import SimpleNamespace

import yaml

from kp.utils.constants import LOGGER


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(v) for v in ns]
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    else:
        return ns

def parse_override_value(key, val):
    # Explicitly handle Boolean parsing for specific keys
    boolean_keys = ["inference_config.smoke_test"]
    
    if key in boolean_keys:
        return val.lower() in ("true", "yes", "1")
    
    # Fallback to YAML parsing for everything else
    return yaml.safe_load(val)

def set_nested(config, key_path, value):
    keys = key_path.split(".")
    for key in keys[:-1]:
        config = config.setdefault(key, {})
    config[keys[-1]] = value


def load_experiment_config(
    experiment_config_path,
    patch_config_path,
    timestamp=None,
    patch_filename=None,
    overrides=None,
):
    with open(experiment_config_path, "r") as f:
        experiment_config = yaml.safe_load(f)
    with open(patch_config_path, "r") as f:
        patch_config = yaml.safe_load(f)

    experiment_config["timestamp"] = timestamp
    experiment_config["patch_config_filename"] = patch_filename
    experiment_config["patch_config"] = patch_config

    LOGGER.info(f"Overrides: {overrides}")
    for item in overrides or []:
        key, val = item.split("=", 1)
        set_nested(experiment_config, key, parse_override_value(key, val))

    return dict_to_namespace(experiment_config)


def load_training_config(training_config_path, overrides=None):
    with open(training_config_path, "r") as f:
        training_config = yaml.safe_load(f)

    LOGGER.info(f"Overrides: {overrides}")
    for item in overrides or []:
        key, val = item.split("=", 1)
        set_nested(training_config, key, yaml.safe_load(val))
    return dict_to_namespace(training_config)


def load_jsonl(file_path):
    """Load a JSONL file and return a list of records."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(file_path, data):
    """Saves a list of dictionaries as JSONL (one JSON object per line)."""
    with open(file_path, "w") as f:
        for entry in data:
            f.write(
                json.dumps(entry) + "\n"
            )  # Write each entry as a JSON object on a new line
