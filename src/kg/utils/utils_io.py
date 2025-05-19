import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from kg.utils.constants import (
    EXPERIMENTS_CONFIG_DIR,
    JOB_CONFIG_DIR,
    LOGGER,
    PATCH_CONFIG_DIR,
)


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
    patch_config_path=None,
    timestamp=None,
    patch_filename=None,
    overrides=None,
):
    with open(experiment_config_path, "r") as f:
        experiment_config = yaml.safe_load(f)

    # Only load patch config if it's not already in the experiment config
    if patch_config_path and "patch_config" not in experiment_config:
        with open(patch_config_path, "r") as f:
            patch_config = yaml.safe_load(f)
        experiment_config["patch_config"] = patch_config
        experiment_config["patch_config_filename"] = patch_filename

    experiment_config["timestamp"] = timestamp

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


def write_yaml(cfg: dict, run_id: str, out_dir: Path = JOB_CONFIG_DIR) -> str:
    """Save dict to <out_dir>/<run_id>.yaml and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = out_dir / f"{run_id}.yaml"
    with cfg_file.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(cfg_file)


def load_patch_config(
    patch_name: str,
    experiments_config_dir=EXPERIMENTS_CONFIG_DIR,
    patch_config_dir: Path = PATCH_CONFIG_DIR,
) -> dict:
    """Load a patch config from the patch config directory.

    Args:
        patch_name: Name of the patch config file
        config_dir: Directory to load the config from. Defaults to PATCH_CONFIG_DIR.

    Returns:
        dict: The loaded patch config
    """
    patch_path = experiments_config_dir / patch_config_dir / patch_name
    with patch_path.open("r") as f:
        return yaml.safe_load(f)


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
            f.write(json.dumps(entry) + "\n")
