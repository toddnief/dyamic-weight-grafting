import itertools
import subprocess
import time

from kg.utils.utils_io import load_patch_config, write_yaml

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")


# Usage: double check all of the ALL_CAPS constants before running

model_dirs = {
    "gemma": {
        "fake_movies_fake_actors": {
            "both": "all_2025-05-03_21-41-43",
            "a2b": "A2B_2025-05-10_03-24-29",
            "b2a": "B2A_2025-05-10_03-24-29",
        },
        "fake_movies_real_actors": {
            "both": "all_2025-05-02_16-30-15",
            "a2b": "A2B_2025-05-09_22-40-14",
            "b2a": "B2A_2025-05-09_22-49-27",
        },
    },
    "gpt2": {
        "fake_movies_fake_actors": {
            "both": "all_2025-05-04_08-41-26",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
        "fake_movies_real_actors": {
            "both": "all_2025-05-04_10-30-33",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
    },
    "llama3": {
        "fake_movies_fake_actors": {
            "both": "all_2025-05-11_18-17-16",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
        "fake_movies_real_actors": {
            "both": "all_2025-05-07_21-51-20",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
    },
    "olmo": {
        "fake_movies_fake_actors": {
            "both": "all_2025-05-06_18-36-58",
            "both_checkpoint": "checkpoint-30800",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
        "fake_movies_real_actors": {
            "both": "all_2025-05-06_18-10-52",
            "both_checkpoint": "checkpoint-35200",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
    },
    "pythia-2.8b": {
        "fake_movies_fake_actors": {
            "both": "all_2025-05-11_18-17-14",
            "both_checkpoint": "checkpoint-26400",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
        "fake_movies_real_actors": {
            "both": "all_2025-05-08_12-10-29",
            "both_checkpoint": "checkpoint-26400",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
    },
    "gpt2-xl": {
        "fake_movies_fake_actors": {
            "both": "all_2025-05-07_22-23-20",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
        "fake_movies_real_actors": {
            "both": "all_2025-05-07_21-56-24",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
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
            "test_sentence_template": "{first_actor} {relation} {relation_preposition} in a movie {preposition}",
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

### SWEEP SETTINGS ###
all_models = ["gemma", "gpt2-xl", "llama3", "pythia-2.8b"]
models_smoke_test = ["gemma"]
models_smoke_test = ["gemma", "gpt2-xl", "llama3", "pythia-2.8b"]

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
SWEEP_PATCH_CONFIG_DIR = "patch_configs"  # Choices: "patch_configs", "patch_configs_lt"

OVERRIDE_PATCH_LAYERS = {
    "first_actor": ["first_quarter", "second_quarter", "third_quarter"],
    "token_idx": ["third_quarter", "fourth_quarter"],
}

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

experiments_dir_addendum = (
    "selective_layers" if OVERRIDE_PATCH_LAYERS != {} else "all_layers"
)


def make_cmd(config_path: str) -> list[str]:
    """Return the make invocation for a given YAML."""
    cmd = ["make", "experiment", f"CONFIG={config_path}"]
    if SINGLE_RUN:
        cmd.append("SINGLE_RUN=0")
    return cmd


for model, dataset_name, patch, lm_head_cfg in itertools.product(
    SWEEP_MODELS, SWEEP_DATASETS, SWEEP_PATCH_CONFIGS, LM_HEAD_CONFIGS
):
    dataset_dir = dataset_dirs[dataset_name]
    donor_model_dir = model_dirs[model][dataset_name]["both"]

    patch_name = patch.split("/")[-1].split(".")[0]

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
    patch_config = load_patch_config(patch, patch_config_dir=SWEEP_PATCH_CONFIG_DIR)

    for patch_target in patch_config.keys():
        if patch_target in OVERRIDE_PATCH_LAYERS:
            patch_config[patch_target]["layers"] = OVERRIDE_PATCH_LAYERS[patch_target]

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
                "both_directions_parent": donor_model_dir,
                "both_directions_checkpoint": (
                    model_dirs[model][dataset_name]["both_checkpoint"]
                    if "both_checkpoint" in model_dirs[model][dataset_name]
                    else None
                ),
                "one_direction_parent": model_dirs[model][dataset_name]["a2b"],
                "one_direction_checkpoint": (
                    model_dirs[model][dataset_name]["a2b_checkpoint"]
                    if "a2b_checkpoint" in model_dirs[model][dataset_name]
                    else None
                ),
            },
            "test_templates": {k: test_templates[k] for k in SELECTED_TEST_TEMPLATES},
            "inference_config": {
                "patch_lm_head": lm_head_cfg,
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

        run_id = (
            f"{timestamp}_{dataset_name}_{model}_{patch_name}_lm_head_{lm_head_cfg}"
        )
        yaml_path = write_yaml(cfg, run_id)

        cmd = make_cmd(yaml_path)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)
