import itertools
import subprocess

smoke_test = "false"  # use "true" or "false"

models = [
    "gemma",
    "llama3",
    "pythia-2.8b",
    "gpt2-xl",
]
datasets = [
    # {"name": "fake_movies_fake_actors", "dir": "2025-05-03_21-10-38"},
    # {"name": "fake_movies_real_actors", "dir": "2025-05-02_16-23-04"},
    # {"name": "real_movies_real_actors_shuffled", "dir": "2025-06-15_13-32-44"},
    {"name": "fake_movies_real_actors_A", "dir": "2025-05-02_16-23-04"},
    {"name": "fake_movies_real_actors_B", "dir": "2025-05-02_16-23-04"},
]
# dataset_types = ["A2B", "B2A"]
dataset_types = ["all"]

for model, dataset, dataset_type in itertools.product(models, datasets, dataset_types):
    dataset_name = dataset["name"]
    dataset_dir = dataset["dir"]

    cmd = [
        "make",
        "train",
        f"SMOKE_TEST={smoke_test}",
        f"MODEL={model}",
        f"DATASET={dataset_name}",
        f"DATASET_DIR={dataset_dir}",
        f"DATASET_TYPE={dataset_type}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)
