import itertools
import subprocess
import time

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

smoke_test = False

# models = ["gemma", "gpt2"]
models = ["gemma"]
datasets = [
    {"name": "fake_movies_fake_actors", "dir": "2025-05-03_21-10-38"},
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
}

patch_directions = ["pre2sft", "sft2pre"]
patch_configs = [
    "first_actor_attn_ffn_all_layers.yaml",
    "preposition_attn_ffn_all_layers.yaml",
    "first_actor_preposition_attn_ffn_all_layers.yaml",
]

for model, dataset, direction, patch in itertools.product(
    models, datasets, patch_directions, patch_configs
):
    dataset_name = dataset["name"]
    dataset_dir = dataset["dir"]
    model_dir = model_dirs[model][dataset_name]

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
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)
