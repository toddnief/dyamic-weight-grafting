import itertools
import subprocess
import time

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

smoke_test = "false"  # Note: use "true" or "false"

# models = ["olmo", "llama3"]
models = ["gemma", "gpt2-xl", "llama3", "pythia-2.8b"] # remove olmo for now
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
    "llama3": {
        # "fake_movies_fake_actors": "all_2025-05-04_10-30-33",
        "fake_movies_real_actors": "all_2025-05-07_21-51-20",
    },
    "olmo": {
        "fake_movies_fake_actors": "all_2025-05-06_18-36-58/checkpoint-30800",
        "fake_movies_real_actors": "all_2025-05-06_18-10-52/checkpoint-35200",
    },
    "pythia-2.8b": {
        # "fake_movies_fake_actors": "all_2025-05-04_10-30-33",
        "fake_movies_real_actors": "all_2025-05-08_12-10-29/checkpoint-26400",
    },
    "gpt2-xl": {
        "fake_movies_fake_actors": "all_2025-05-07_22-23-20",
        "fake_movies_real_actors": "all_2025-05-07_21-56-24",
    },
}

patch_configs = [
    "no_patching.yaml",
    "fe.yaml",
    "lt.yaml",
    "fe_lt.yaml",
    "fe_lt_complement.yaml",
    "not_lt.yaml",
    "m.yaml",
    "fe_m.yaml",
    "fe_m_lt.yaml",
    "m_lt.yaml",
    "not_fe_m.yaml",
    "not_fe_m_lt.yaml",
    "fe_m_lt_p.yaml",
    "fe_m_p.yaml",
]


def create_command(model, patch, direction, dataset_name, dataset_dir, model_dir):
    return [
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


for model, dataset, patch in itertools.product(models, datasets, patch_configs):
    if (model == "pythia-2.8b" or model == "llama3") and dataset[
        "name"
    ] == "fake_movies_fake_actors":
        continue

    if patch == "no_patch.yaml":
        direction = "pre2sft"
    else:
        direction = "sft2pre"

    dataset_name = dataset["name"]
    dataset_dir = dataset["dir"]
    model_dir = model_dirs[model][dataset_name]

    if patch == "no_patch.yaml":
        for direction in ["pre2sft", "sft2pre"]:
            cmd = create_command(
                model, patch, direction, dataset_name, dataset_dir, model_dir
            )
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)
    else:
        direction = "sft2pre"
        cmd = create_command(
            model, patch, direction, dataset_name, dataset_dir, model_dir
        )
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)
