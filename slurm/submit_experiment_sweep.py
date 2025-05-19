import itertools
import subprocess
import time
from pathlib import Path

import yaml

from kg.utils.constants import EXPERIMENTS_CONFIG_DIR, PATCH_CONFIG_DIR

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")


CFG_DIR = EXPERIMENTS_CONFIG_DIR / "generated_cfgs"


# TODO: Put this in utils
def write_yaml(cfg: dict, run_id: str, out_dir: Path = CFG_DIR) -> str:
    """Save dict to <out_dir>/<run_id>.yaml and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = out_dir / f"{run_id}.yaml"
    with cfg_file.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(cfg_file)


# TODO: Put this in utils
def load_patch_config(patch_name: str) -> dict:
    """Load a patch config from the patch config directory."""
    patch_path = PATCH_CONFIG_DIR / patch_name
    with patch_path.open("r") as f:
        return yaml.safe_load(f)


def make_cmd(config_path: str) -> list[str]:
    """Return the make invocation for a given YAML."""
    return [
        "make",
        "experiment",
        f"CONFIG={config_path}",
        "PATCH_CONFIG=",  # Empty patch config since it's included in the experiment config
    ]


# Usage: double check all of the ALL_CAPS constants before running

model_dirs = {
    "gemma": {
        "fake_movies_fake_actors": "all_2025-05-03_21-41-43",
        "fake_movies_real_actors": "all_2025-05-02_16-30-15",
    },
    "gpt2": {
        "fake_movies_fake_actors": "all_2025-05-04_08-41-26",
        "fake_movies_real_actors": "all_2025-05-04_10-30-33",
    },
    "llama3": {
        "fake_movies_fake_actors": "all_2025-05-11_18-17-16",
        "fake_movies_real_actors": "all_2025-05-07_21-51-20",
    },
    "olmo": {
        "fake_movies_fake_actors": "all_2025-05-06_18-36-58/checkpoint-30800",
        "fake_movies_real_actors": "all_2025-05-06_18-10-52/checkpoint-35200",
    },
    "pythia-2.8b": {
        "fake_movies_fake_actors": "all_2025-05-11_18-17-14/checkpoint-26400",
        "fake_movies_real_actors": "all_2025-05-08_12-10-29/checkpoint-26400",
    },
    "gpt2-xl": {
        "fake_movies_fake_actors": "all_2025-05-07_22-23-20",
        "fake_movies_real_actors": "all_2025-05-07_21-56-24",
    },
}

dataset_dirs = {
    "fake_movies_fake_actors": "2025-05-03_21-10-38",
    "fake_movies_real_actors": "2025-05-02_16-23-04",
}

# Add dataset name to test sentence templates
dataset2test_templates = {
    "fake_movies_fake_actors": {
        "sentence_1": {
            "test_sentence_template": "{first_actor} {relation} {relation_preposition} {movie_title} {preposition}",
            "preposition": "alongside",
            "relation": "stars",
            "relation_preposition": "in",
        },
        "sentence_2": {
            "test_sentence_template": "Q: Who {relation} {relation_preposition} a movie {preposition} {first_actor}? A: An actor named",
            "preposition": "with",
            "relation": "stars",
            "relation_preposition": "in",
        },
        "sentence_3": {
            "test_sentence_template": "In a new film, {first_actor} {relation} {relation_preposition} {movie_title} {preposition} the other lead actor, whose name is:",
            "preposition": "with",
            "relation": "appears",
            "relation_preposition": "in",
        },
    }
}

dataset_target_keys = {
    "fake_movies_fake_actors": "second_actor",
    "fake_movies_real_actors": "second_actor",
}

### RUN SETTINGS THAT DON'T CHANGE ###
TOP_K = 20
DROPOUT_RATE = 0.0
DROPOUT_UNIT = "layer"
DROPOUT_STRATEGY = "count"

### RUN SETTINGS TO CHANGE ###
SMOKE_TEST = True
SINGLE_RUN = True
REVERSAL = False  # Note: this runs the "reversal" experiment — both2one patches to A2B
OVERRIDE_LAYERS = True  # Uses the override layers option when patching (rather than the layers in the config)

### SWEEP SETTINGS ###
all_models = ["gemma", "gpt2-xl", "llama3", "pythia-2.8b"]
models_smoke_test = ["gemma"]
# Update this
SWEEP_MODELS = ["gemma"]

main_patch_configs = [
    "no_patching.yaml",  # baseline
    "fe.yaml",
    "lt.yaml",
    "fe_lt.yaml",
    "fe_lt_complement.yaml",
    "not_fe.yaml",
    "not_lt.yaml",
]
component_patch_configs = [
    "no_patching.yaml",
    "attn_ffn.yaml",
    "attn_o.yaml",
    "attn_o_ffn.yaml",
    "o.yaml",
    "o_ffn.yaml",
    "o_ffn_up.yaml",
    "o_ffn_down.yaml",
    "ffn.yaml",
]
patch_configs_smoke_test = ["no_patching.yaml", "fe.yaml"]
# Update this
SWEEP_PATCH_CONFIGS = main_patch_configs

# Update this
all_datasets = ["fake_movies_fake_actors", "fake_movies_real_actors"]
SWEEP_DATASETS = ["fake_movies_real_actors"]

lm_head_configs = ["never", "always", "last_token"]
lm_head_configs_smoke_test = ["never"]
# Update this
LM_HEAD_CONFIGS = ["never"]

# Update this
SELECTED_TEST_TEMPLATES = ["sentence_1", "sentence_2", "sentence_3"]

### Settings logic ###
patch_direction_default = "both2one" if REVERSAL else "sft2pre"

if SMOKE_TEST:
    SWEEP_MODELS = models_smoke_test
    SWEEP_PATCH_CONFIGS = patch_configs_smoke_test
    LM_HEAD_CONFIGS = lm_head_configs_smoke_test

experiments_dir_addendum = "selective_layers" if OVERRIDE_LAYERS else "all_layers"


counter = 0
for model, dataset_name, patch, lm_head_cfg in itertools.product(
    SWEEP_MODELS, SWEEP_DATASETS, SWEEP_PATCH_CONFIGS, LM_HEAD_CONFIGS
):
    counter += 1
    dataset_dir = dataset_dirs[dataset_name]
    model_dir = model_dirs[model][dataset_name]

    # Note: same test templates for both datasets so override
    test_template_name = (
        "fake_movies_fake_actors"
        if dataset_name == "fake_movies_real_actors"
        else dataset_name
    )
    test_templates = dataset2test_templates[test_template_name]

    directions = (
        ["pre2sft", "sft2pre"]
        if patch == "no_patching.yaml"
        else [patch_direction_default]
    )

    # Load the patch config
    patch_config = load_patch_config(patch)

    for direction in directions:
        cfg = {
            "smoke_test": SMOKE_TEST,
            "patching_flag": patch != "no_patching.yaml",
            "model": {
                "pretrained": model,
                "patch_direction": direction,
            },
            "paths": {
                "experiments_dir_addendum": experiments_dir_addendum,
                "dataset_name": dataset_name,
                "dataset_dir": dataset_dir,
                "both_directions_parent": model_dir,
                "both_directions_checkpoint": None,
                # The one-direction fields are optional; include only when relevant.
                **(
                    {
                        "one_direction_parent": model_dir,
                        "one_direction_checkpoint": None,
                    }
                    if direction in ("pre2sft", "sft2pre")
                    else {}
                ),
            },
            "test_templates": {k: test_templates[k] for k in SELECTED_TEST_TEMPLATES},
            "inference_config": {
                "patch_lm_head": lm_head_cfg,
                "override_layers": OVERRIDE_LAYERS,
                "dropout_rate": DROPOUT_RATE,
                "dropout_unit": DROPOUT_UNIT,  # layer | matrix
                "dropout_strategy": DROPOUT_STRATEGY,  # count | random
            },
            "analysis_config": {
                "top_k": TOP_K,
                "target_key": dataset_target_keys[dataset_name],
            },
            "patch_config": patch_config,
            "patch_config_filename": patch,
        }

        run_id = f"{timestamp}_{counter:03d}"
        yaml_path = write_yaml(cfg, run_id)

        cmd = make_cmd(yaml_path)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)
