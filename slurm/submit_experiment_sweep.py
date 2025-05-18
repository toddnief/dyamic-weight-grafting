import itertools
import subprocess
import time

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

smoke_test = "false"  # Note: use "true" or "false" for Makefile / slurm
single_run = True  # Use booleans here
reversal = False

if reversal:
    patch_direction = "both2one"
    # patch_direction = "sft2pre"  # Hack to schedule a baseline run
else:
    patch_direction = "sft2pre"

override_layers = True


models_smoke_test = ["gemma"]
models = ["gemma", "gpt2-xl", "llama3", "pythia-2.8b"]
# models = ["gpt2-xl"]

datasets = [
    # {"name": "fake_movies_fake_actors", "dir": "2025-05-03_21-10-38"},
    {"name": "fake_movies_real_actors", "dir": "2025-05-02_16-23-04"},
]

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

# These are the baseline patch configs
patch_configs = [
    # BASELINE PATCH
    "no_patching.yaml",
    # FE + LT
    "fe.yaml",
    "lt.yaml",
    "fe_lt.yaml",
    # RELATION PATCHES
    # "r.yaml",
    # "fe_r.yaml",
    # "r_lt.yaml",
    # "fe_r_lt.yaml",
    # COMPLEMENT PATCHES
    "fe_lt_complement.yaml",
    "not_fe.yaml",
    "not_lt.yaml",
    # MOVIE PATCHES
    # "m.yaml",
    # "fe_m.yaml",
    # "fe_m_lt.yaml",
    # "m_lt.yaml",
    # "not_fe_m.yaml",
    # "not_fe_m_lt.yaml",
    # EXTRA
    # "fe_m_p_lt.yaml",
    # "fe_m_p.yaml",
    # "r_rp.yaml",
    # "r_rp_lt.yaml",
    # "rp.yaml",
]

# patch_configs = [
#     "no_patching.yaml",
#     "attn_ffn.yaml",
#     "attn_o.yaml",
#     "attn_o_ffn.yaml",
#     "o.yaml",
#     "o_ffn.yaml",
#     "o_ffn_up.yaml",
#     "o_ffn_down.yaml",
#     "ffn.yaml",
# ]

patch_configs_smoke_test = [
    # BASELINE PATCH
    "no_patching.yaml",
    # FE + LT
    "fe.yaml",
]

lm_head_configs_smoke_test = ["never"]
# lm_head_configs = ["always", "never", "last_token"]
lm_head_configs = ["never"]

if smoke_test == "true":
    models = models_smoke_test
    # patch_configs = patch_configs_smoke_test
    lm_head_configs = lm_head_configs_smoke_test

# TODO: Hacky...
experiments_dir_addendum = "selective_layers" if override_layers else "all_layers"
# experiments_dir_addendum = "lt_reversal_baseline_top_half"


def create_command(
    model,
    patch,
    direction,
    dataset_name,
    dataset_dir,
    model_dir,
    lm_head_config,
    single_run=False,
    override_layers=False,
    experiments_dir_addendum=None,
    reversal=False,
):
    cmd = [
        "make",
        "experiment",
        f"SMOKE_TEST={smoke_test}",
        f"TIMESTAMP={timestamp}",
        f"PATCH_CONFIG={patch}",
        f"MODEL={model}",
        f"DATASET={dataset_name}",
        f"DATASET_DIR={dataset_dir}",
        f"MODEL_DIR={model_dir}",
        f"DIRECTION={direction}",
        f"LM_HEAD_CONFIG={lm_head_config}",
    ]
    if experiments_dir_addendum:
        cmd.append(f"EXPERIMENTS_DIR_ADDENDUM={experiments_dir_addendum}")
    # Note: This passes a dropout index to the script - 0 corresponds to no dropout
    if single_run:
        cmd.append("SINGLE_RUN=0")
    # Add override layers boolean
    if override_layers:
        cmd.append("OVERRIDE_LAYERS=1")
    if reversal:
        cmd.append("CONFIG=config_experiments_reversal.yaml")
    return cmd


for model, dataset, patch, lm_head_config in itertools.product(
    models, datasets, patch_configs, lm_head_configs
):
    dataset_name = dataset["name"]
    dataset_dir = dataset["dir"]
    model_dir = model_dirs[model][dataset_name]

    if patch == "no_patching.yaml":
        for direction in ["pre2sft", "sft2pre"]:
            cmd = create_command(
                model,
                patch,
                direction,
                dataset_name,
                dataset_dir,
                model_dir,
                lm_head_config,
                single_run,
                override_layers,
                experiments_dir_addendum,
                reversal,
            )
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)
    else:
        direction = patch_direction
        cmd = create_command(
            model,
            patch,
            direction,
            dataset_name,
            dataset_dir,
            model_dir,
            lm_head_config,
            single_run,
            override_layers,
            experiments_dir_addendum,
            reversal,
        )
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)
