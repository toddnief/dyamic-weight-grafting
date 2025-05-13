from datetime import datetime
import json
from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from kp.utils.constants import FIGURES_DIR

def find_results_files(base_dir: Path | str, allow_smoke_test: bool = False):
    """
    Collect every results.json under `base_dir`, applying only the directory
    filters (archive / bug / skip / smoke_test)
    """
    base_dir = Path(base_dir)

    results = []
    for path in base_dir.rglob("results.json"):
        parts = path.parts
        if any(sub in part for part in parts for sub in ("archive", "bug", "skip")):
            continue
        if not allow_smoke_test and any("smoke_test" in p for p in parts):
            continue
        results.append(path)

    print(f"Found {len(results)} 'results.json' files.")
    return results


def parse_path(results_file_path: Path, base_dir: Path):
    """
    Parses the file path to extract experiment metadata.
    Expected path structure relative to base_dir:
    lm_head_setting/dataset/model/patch_direction/patch_type/run_id/sentence_id/dropout_rate/results.json
    """
    # Ensure both are Path objects
    if not isinstance(results_file_path, Path):
        results_file_path = Path(results_file_path)
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)

    try:
        # Ensure results file is within the base directory
        if base_dir not in results_file_path.parents:
            print(f"Warning: File {results_file_path} is not under base_dir {base_dir}")
            return None

        # Compute the relative path
        relative_path = results_file_path.relative_to(base_dir)
        components = list(relative_path.parts)

        if len(components) == 9 and components[-1] == "results.json":
            (
                lm_head_setting,
                dataset,
                model,
                patch_direction,
                patch_type,
                run_id,
                sentence_id,
                dropout_rate,
                _,
            ) = components

            return {
                "lm_head_setting": lm_head_setting,
                "dataset": dataset,
                "model": model,
                "patch_direction": patch_direction,
                "patch_type": patch_type,
                "run_id": run_id,
                "sentence_id": sentence_id,
                "dropout_rate": dropout_rate,
                "full_path": str(results_file_path),  # Store string path
            }
        else:
            print(
                f"Warning: Path structure mismatch for {results_file_path}. Relative: '{relative_path}', Components: {len(components)} {components}"
            )
            return None
    except Exception as e:
        print(f"Error parsing path {results_file_path}: {e}")
        return None


def calculate_metrics_from_file(results_json_path, top_k=5):
    """
    Reads a results.json file and calculates metrics.
    Metrics: mean target rank, top-5 accuracy, mean target probability.
    Assumes target token rank is 1-indexed for top-5 accuracy (rank <= 5).
    """
    try:
        with open(results_json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing {results_json_path}: {e}")
        return None  # Indicates a file read/parse error

    if "results" not in data or not isinstance(data["results"], list):
        # print(f"Warning: 'results' key missing or not a list in {results_json_path}")
        return {  # Return NaNs if structure is invalid but file was readable
            "mean_target_rank": float("nan"),
            "top_5_accuracy": float("nan"),
            "mean_target_prob": float("nan"),
        }

    if not data["results"]:  # Empty list of results
        return {
            "mean_target_rank": float("nan"),
            "top_5_accuracy": float("nan"),
            "mean_target_prob": float("nan"),
        }

    target_ranks = []
    is_in_top_k = []
    target_probs = []

    for res_item in data["results"]:
        if "target" in res_item and isinstance(res_item["target"], dict):
            target_info = res_item["target"]

            if "token_rank" in target_info and isinstance(
                target_info["token_rank"], (int, float)
            ):
                rank = target_info["token_rank"] + 1  # Note: token_rank is 0-indexed
                target_ranks.append(rank)
                is_in_top_k.append(
                    1 if rank <= top_k and rank >= 1 else 0
                )  # Ensure rank is positive
            else:
                target_ranks.append(float("nan"))
                is_in_top_k.append(float("nan"))

            if "token_prob" in target_info and isinstance(
                target_info["token_prob"], (int, float)
            ):
                target_probs.append(target_info["token_prob"])
            else:
                target_probs.append(float("nan"))
        else:  # Target info missing for a result item
            target_ranks.append(float("nan"))
            is_in_top_k.append(float("nan"))
            target_probs.append(float("nan"))

    mean_rank = (
        np.nanmean(target_ranks)
        if any(not np.isnan(r) for r in target_ranks)
        else float("nan")
    )
    top_k_acc = (
        np.nanmean(is_in_top_k)
        if any(not np.isnan(r) for r in is_in_top_k)
        else float("nan")
    )
    mean_prob = (
        np.nanmean(target_probs)
        if any(not np.isnan(r) for r in target_probs)
        else float("nan")
    )

    return {
        "mean_target_rank": mean_rank,
        "top_k_accuracy": top_k_acc,
        "mean_target_prob": mean_prob,
    }

from collections import defaultdict
from pathlib import Path
import numpy as np

# pre‑compile for speed
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

def parse_timestamp(dir_name: str):
    """
    Extract the latest timestamp from a directory / run‑id string.
    Expected pattern: YYYY-MM-DD_HH-MM-SS (may appear multiple times).
    Returns a datetime object or None if no timestamp is found.
    """
    matches = _TS_RE.findall(dir_name)
    if not matches:
        return None
    return max(datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S") for ts in matches)

def organize_results(all_results_files, base_dir: Path):
    """
    Build nested dict:
        organized[dataset][lm_head_setting][model][sentence_id][patch] = metrics_dict
    and keep only the *newest* run for every (dataset, lm_head, model, sentence, patch) combo.
    """
    organized = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    parsed_ok, metrics_ok = 0, 0

    for fp in all_results_files:
        info = parse_path(fp, base_dir)
        if not info:
            continue
        parsed_ok += 1

        metrics = calculate_metrics_from_file(fp)
        if metrics is None:
            print(f"Skipping unreadable {fp}")
            continue
        metrics_ok += 1

        dset = info["dataset"]
        lm_head = info["lm_head_setting"]
        model = info["model"]
        sent  = info["sentence_id"]
        patch = info["patch_type"]

        if patch == "no_patching":
            patch = (
                "no_patching_sft2pre" if "sft2pre" in fp.parts else
                "no_patching_pre2sft" if "pre2sft" in fp.parts else
                patch
            )

        ts = parse_timestamp(info["run_id"])  # newest run wins
        slot = organized[dset][lm_head][model][sent]

        if (
            patch not in slot or
            ts and (slot[patch]["timestamp"] is None or ts > slot[patch]["timestamp"])
        ):
            slot[patch] = {"metrics": metrics, "timestamp": ts}

    for d in organized.values():
        for l in d.values():
            for m in l.values():
                for s in m.values():
                    for p in list(s.keys()):
                        s[p] = s[p]["metrics"]

    print(f"Attempted to parse {len(all_results_files)} files.")
    print(f"Successfully parsed {parsed_ok} paths and calculated metrics for {metrics_ok}.")
    print(f"Organized data into {len(organized)} datasets.")
    return organized


# Setup order and display names for patch configs
PATCH_MAPPING = {
    "no_patching_pre2sft": ("baseline", "SFT"),
    "no_patching_sft2pre": ("baseline", "PRE"),
    "fe": ("single_token", "FE"),
    "lt": ("single_token", "LT"),
    "fe_lt": ("multi_token", "FE+LT"),
    "r": ("single_token", "R"),
    "fe_r": ("multi_token", "FE+R"),
    "r_lt": ("multi_token", "R+LT"),
    "fe_r_lt": ("multi_token", "FE+R+LT"),
    "fe_lt_complement": ("complement", "(FE+LT)^C"),
    "not_lt": ("complement", "NOT LT"),
    "m": ("single_token", "M"),
    "fe_m": ("multi_token", "FE+M"),
    "fe_m_lt": ("multi_token", "FE+M+LT"),
    "m_lt": ("multi_token", "M+LT"),
    "not_fe_m": ("complement", "NOT FE+M"),
    "not_fe_m_lt": ("complement", "NOT FE+M+LT"),
}

# Define the order for the buckets
BUCKET_ORDER = {
    "baseline": 0,
    "single_token": 1,
    "multi_token": 2,
    "complement": 3
}

# Skip these patch configs
SKIP_SET = {"r_rp", "r_rp_lt", "rp", "rp_lt"}

DEFAULT_BUCKET = "unknown"
DEFAULT_ORDER = 99

def get_patch_order_and_name(patch_name):
    if patch_name in PATCH_MAPPING:
        bucket, display_name = PATCH_MAPPING[patch_name]
        order = BUCKET_ORDER.get(bucket, DEFAULT_ORDER)
        return order, display_name
    
    return DEFAULT_ORDER, patch_name

def plot_metric(organized_data, metric_key, layers_setting=None, save=False, save_dir=FIGURES_DIR):
    """
    Generates bar plots for a specified metric across patch configurations,
    grouped by dataset, sentence, and model (in that order).

    Args:
        organized_data (dict): Nested as
            organized_data[dataset][model][sentence][patch] = metrics_dict
        metric_key (str): Metric key to plot
    """
    if not organized_data:
        print("No data available to plot.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    metric_config = {
        "top_k_accuracy": {"label": "Top-K Accuracy", "color": "viridis"},
        "mean_target_prob": {"label": "Mean Target Probability", "color": "plasma"},
        "mean_target_rank": {"label": "Mean Target Rank", "color": "cividis"},
    }
    if metric_key not in metric_config:
        raise ValueError(
            f"Metric '{metric_key}' is not valid. Choose from {list(metric_config.keys())}."
        )

    cfg = metric_config[metric_key]

    # Dataset → Sentence → Model
    for dataset_name, lm_head_settings in organized_data.items():
        for lm_head_setting, models_data in lm_head_settings.items():
            sentences = sorted(
                {s for m in models_data.values() for s in m}
            )  # all sentences present
            for sentence_id in sentences:
                for model_name, sentences_data in models_data.items():
                    if sentence_id not in sentences_data:
                        continue
                    patch_config_results = sentences_data[sentence_id]

                    if not patch_config_results:
                        print(
                            f"Skipping {dataset_name} / {sentence_id} / {model_name}: No patch data."
                        )
                        continue
                    
                    patch_names, metric_values = [], []

                    # Sort first by order bucket, then alphabetically within the bucket
                    sorted_patches = sorted(
                        patch_config_results.items(),
                        key=lambda x: get_patch_order_and_name(x[0])
                    )

                    # Collect the display names and metric values
                    seen_display_names = set()
                    for patch_name, metrics in sorted_patches:
                        if patch_name in SKIP_SET:
                            continue
                        if metric_key in metrics and not np.isnan(metrics[metric_key]):
                            _, display_name = get_patch_order_and_name(patch_name)
                            
                            # Ensure uniqueness by appending index if a duplicate is found
                            if display_name in seen_display_names:
                                counter = 1
                                new_display_name = f"{display_name}_{counter}"
                                while new_display_name in seen_display_names:
                                    counter += 1
                                    new_display_name = f"{display_name}_{counter}"
                                display_name = new_display_name
                            
                            seen_display_names.add(display_name)
                            patch_names.append(display_name)
                            metric_values.append(metrics[metric_key])

                    if not patch_names:
                        print(
                            f"No valid data for {metric_key} in {dataset_name} / {sentence_id} / {model_name}"
                        )
                        continue

                    plt.figure(figsize=(max(10, len(patch_names) * 0.8), 7))
                    colors = plt.cm.get_cmap(cfg["color"])(
                        np.linspace(0, 1, len(patch_names))
                    )
                    bars = plt.bar(patch_names, metric_values, color=colors)

                    # Define title mapping
                    metric_title_mapping = {
                        "top_k_accuracy": "Top-5 Accuracy",
                        "mean_target_prob": "Mean Target Probability",
                        "mean_target_rank": "Mean Target Rank",
                    }

                    model_title_mapping = {
                        "gpt2-xl": "GPT-2 XL",
                        "gemma": "Gemma-1.1-2B-IT",
                        "olmo": "OLMo-1B",
                        "llama3": "Llama-3.2-1B",
                        "pythia-2.8b": "Pythia-2.8B",
                        "gpt2": "GPT-2",
                    }

                    model_sentence_mapping = {
                        "sentence_1": "Sentence 1",
                        "sentence_2": "Sentence 2",
                        "sentence_3": "Sentence 3",
                    }

                    dataset_title_mapping = {
                        "fake_movies_real_actors": "Fake Movies, Real Actors",
                        "fake_movies_fake_actors": "Fake Movies, Fake Actors",
                    }

                    lm_head_title_mapping = {
                        "lm_head_always": "LM Head: Always",
                        "lm_head_never": "LM Head: Never",
                        "lm_head_last_token": "LM Head: Last Token",
                    }

                    layers_title_mapping = {
                        "all_layers": "All Layers",
                        "selective_layers": "Selective Layers",
                    }

                    title = (
                        f"{metric_title_mapping[metric_key]}\n"
                        f"{model_title_mapping[model_name]}"
                        f" | {model_sentence_mapping[sentence_id]}"
                        f" | {dataset_title_mapping[dataset_name]}"
                        f" | {lm_head_title_mapping[lm_head_setting]}"
                        f" | {layers_title_mapping[layers_setting]}"
                    )

                    plt.title(
                        title,
                        fontsize=14,
                    )
                    plt.xlabel("Patch Configuration", fontsize=12)
                    plt.ylabel(cfg["label"], fontsize=12)
                    plt.xticks(rotation=90, ha="right")
                    plt.grid(axis="y", linestyle="--", alpha=0.7)

                    for bar in bars:
                        yval = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width() / 2,
                            yval,
                            f"{yval:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                    plt.tight_layout()

                    if save:
                        stamp = datetime.datetime.now().strftime("%Y%m%d‑%H%M%S")
                        fname = (
                            f"{metric_key}_{dataset_name}_sent{sentence_id}_"
                            f"{model_name}" + f"_{lm_head_setting}" + f"_{stamp}.png"
                        )
                        plt.savefig(save_dir / fname, dpi=300, bbox_inches="tight")

                    plt.show()
                    plt.close()
