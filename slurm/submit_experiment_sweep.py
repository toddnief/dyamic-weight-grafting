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
            "a2b": "A2B_2025-05-10_02-56-17",
            "b2a": "B2A_2025-05-10_03-00-47",
        },
        "fake_movies_real_actors": {
            "both": "all_2025-05-02_16-30-15",
            "a2b": "A2B_2025-05-10_03-24-29",
            "b2a": "B2A_2025-05-10_03-24-29",
        },
        "real_movies_real_actors": {
            "both": "all_2025-05-02_16-30-15",
            "a2b": "A2B_2025-05-10_03-24-29",
            "b2a": "B2A_2025-05-10_03-24-29",
        },
        "counterfact": {
            "both": "all_2025-06-08_11-41-09",
        },
    },
    "gpt2": {
        "fake_movies_fake_actors": {
            "both": "all_2025-05-04_08-41-26",
            "a2b": "A2B_2025-04-26_21-17-38",
            "b2a": "B2A_2025-04-26_21-20-11",
        },
        "fake_movies_real_actors": {
            "both": "all_2025-05-04_10-30-33",
            "a2b": "A2B_2025-04-26_21-42-22",
            "b2a": "B2A_2025-04-26_21-46-23",
        },
        "real_movies_real_actors": {
            "both": "all_2025-05-04_10-30-33",
            "a2b": "A2B_2025-04-26_21-42-22",
            "b2a": "B2A_2025-04-26_21-46-23",
        },
    },
    "llama3": {
        "fake_movies_fake_actors": {
            "both": "all_2025-05-11_18-17-16",
            "a2b": "A2B_2025-05-09_22-38-44",
            "b2a": "B2A_2025-05-09_22-38-44",
        },
        "fake_movies_real_actors": {
            "both": "all_2025-05-07_21-51-20",
            "a2b": "A2B_2025-05-09_22-40-14",
            "b2a": "B2A_2025-05-09_22-49-27",
        },
        "real_movies_real_actors": {
            "both": "all_2025-05-07_21-51-20",
            "a2b": "A2B_2025-05-09_22-40-14",
            "b2a": "B2A_2025-05-09_22-49-27",
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
        "real_movies_real_actors": {
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
            # "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-19_17-46-35",
        },
        "fake_movies_real_actors": {
            "both": "all_2025-05-08_12-10-29",
            "both_checkpoint": "checkpoint-26400",
            # "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-22_11-18-57",
        },
        "real_movies_real_actors": {
            "both": "all_2025-05-08_12-10-29",
            "both_checkpoint": "checkpoint-26400",
            "b2a": "B2A_2025-05-22_11-18-57",
        },
        "counterfact": {
            "both": "A2B_2025-05-27_15-06-21",  # Naming convention here is off since model already "knows" the info
            "both_checkpoint": "checkpoint-26304",
            "b2a": None,
        },
    },
    "gpt2-xl": {
        "fake_movies_fake_actors": {
            "both": "all_2025-05-07_22-23-20",
            "a2b": "A2B_2025-05-09_22-13-04",
            "b2a": "B2A_2025-05-09_22-13-41",
        },
        "fake_movies_real_actors": {
            "both": "all_2025-05-07_21-56-24",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
        "real_movies_real_actors": {
            "both": "all_2025-05-07_21-56-24",
            "a2b": "A2B_2025-05-09_22-34-37",
            "b2a": "B2A_2025-05-09_22-34-34",
        },
    },
}

dataset_dirs = {
    "fake_movies_fake_actors": "2025-05-03_21-10-38",
    "fake_movies_real_actors": "2025-05-02_16-23-04",
    "real_movies_real_actors": "2025-05-26_11-58-04",
    "counterfact": None,  # counterfact is downloaded from huggingface
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
            "test_sentence_template": "{first_actor} {relation} {relation_preposition} in {movie_title} {preposition}",
            "preposition": "alongside",
            "relation": "stars",
            "relation_preposition": "in",
        },
        "sentence_4": {
            "test_sentence_template": "In a new film, {first_actor} {relation} {relation_preposition} {movie_title} {preposition} the other lead actor, whose name is:",
            "preposition": "with",
            "relation": "appears",
            "relation_preposition": "in",
        },
    },
    "real_movies_real_actors": {
        "sentence_1": {
            "test_sentence_template": "In a new film, {second_actor} {relation} {relation_preposition} {movie_title} {preposition} the other lead actor, whose name is:",
            "preposition": "with",
            "relation": "appears",
            "relation_preposition": "in",
        }
    },
    "counterfact": "counterfact_sentence",  # counterfact builds test sentences directly from the example
}

dataset_target_keys = {
    "fake_movies_fake_actors": "second_actor",
    "fake_movies_real_actors": "second_actor",
    "real_movies_real_actors": "first_actor",
    "counterfact": "subject",
}

### RUN SETTINGS THAT DON'T CHANGE ###
TOP_K = 20
DROPOUT_RATE = 0.0
DROPOUT_UNIT = "layer"
DROPOUT_STRATEGY = "count"

### RUN SETTINGS TO CHANGE ###
SMOKE_TEST = False
SINGLE_RUN = True
REVERSAL = False  # Note: this runs the "reversal" experiment — both2one patches to B2A
N_EXAMPLES = 1000

### SWEEP SETTINGS ###
# Update this
# all_datasets: ["fake_movies_fake_actors", "fake_movies_real_actors", "real_movies_real_actors", "counterfact"]
SWEEP_DATASETS = ["counterfact"]

# Update this
# all_models: ["gemma", "gpt2-xl", "llama3", "pythia-2.8b"]
SWEEP_MODELS = ["gemma"]
models_smoke_test = ["gemma"]

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
# Check this
SWEEP_PATCH_CONFIGS = main_patch_configs if not REVERSAL else component_patch_configs

sweep_patch_config_dir = "patch_configs" if not REVERSAL else "patch_configs_lt"

# Ugly hack to run counterfact experiments
if SWEEP_DATASETS == ["counterfact"]:
    sweep_patch_config_dir = "patch_configs_cf"

# Update this
OVERRIDE_PATCH_LAYERS_BOOLEAN = False
OVERRIDE_PATCH_LAYERS = {
    # "first_actor": [
    #     "first_quarter",
    #     "second_quarter",
    #     "third_quarter",
    #     "fourth_quarter",
    # ],
    "token_idx": ["third_quarter", "fourth_quarter"],
}

lm_head_configs = ["never", "always", "last_token"]
lm_head_configs_smoke_test = ["never"]
# Update this
LM_HEAD_CONFIGS = ["never"]

# Update this
# all_test_templates: ["sentence_1", "sentence_2", "sentence_3"]
SELECTED_TEST_TEMPLATES = ["sentence_1"]

### Settings logic ###
patch_direction_default = "both2one" if REVERSAL else "sft2pre"

# Check this if running a smoke test
if SMOKE_TEST:
    SWEEP_MODELS = models_smoke_test
    SWEEP_PATCH_CONFIGS = patch_configs_smoke_test
    LM_HEAD_CONFIGS = lm_head_configs_smoke_test

experiments_dir_addendum = (
    "selective_layers" if OVERRIDE_PATCH_LAYERS_BOOLEAN else "all_layers"
)
if REVERSAL:
    experiments_dir_addendum = f"{experiments_dir_addendum}_reversal"
if SMOKE_TEST:
    experiments_dir_addendum = f"{experiments_dir_addendum}_smoke_test"


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
    patch_config = load_patch_config(patch, patch_config_dir=sweep_patch_config_dir)

    for patch_target in patch_config.keys():
        if patch_target in OVERRIDE_PATCH_LAYERS and OVERRIDE_PATCH_LAYERS_BOOLEAN:
            patch_config[patch_target]["layers"] = OVERRIDE_PATCH_LAYERS[patch_target]

    for direction in directions:
        if test_templates is type(dict):
            test_templates = {k: test_templates[k] for k in SELECTED_TEST_TEMPLATES}
        cfg = {
            "smoke_test": SMOKE_TEST,
            "patching_flag": patch != "no_patching.yaml",
            "n_examples": N_EXAMPLES if dataset_name == "counterfact" else None,
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
                "one_direction_parent": (
                    model_dirs[model][dataset_name]["b2a"]
                    if "b2a" in model_dirs[model][dataset_name]
                    else None
                ),
                "one_direction_checkpoint": (
                    model_dirs[model][dataset_name]["b2a_checkpoint"]
                    if "b2a_checkpoint" in model_dirs[model][dataset_name]
                    else None
                ),
            },
            "test_templates": test_templates,
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

        run_id = f"{timestamp}_{dataset_name}_{model}_{patch_name}_{direction}_lm_head_{lm_head_cfg}"
        yaml_path = write_yaml(cfg, run_id)

        cmd = make_cmd(yaml_path)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)
