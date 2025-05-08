import argparse
import json
from pathlib import Path
from statistics import mean, variance
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from kp.scripts.run_experiment import get_experiment_timestamp_dir
from kp.utils.constants import (
    EXPERIMENTS_CONFIG_DIR,
    LOGGER,
    PATCH_CONFIG_DIR,
    TIMESTAMP,
)
from kp.utils.utils_io import load_experiment_config


def load_experiment_results(
    results_dir: Path, top_k: int = 20
) -> tuple[List[Dict[str, Any]], Dict[float, List[Dict[str, Any]]]]:
    results_paths = [p for p in results_dir.rglob("*.json") if "figures" not in p.parts]
    if not results_paths:
        LOGGER.warning(f"No JSON results found in {results_dir}")
        return [], {}

    poor_performance_examples = {}
    results = []

    for results_path in results_paths:
        with open(results_path, "r") as f:
            data = json.load(f)

        try:
            dropout_rate = data["inference_config"].get("dropout_rate")
        except KeyError:
            raise ValueError(f"No dropout rate found for {results_path}")

        target_probs = [
            ex["target"]["token_prob"]
            for ex in data["results"]
            if "target" in ex and "token_prob" in ex["target"]
        ]

        accuracy = [
            int(ex["target"]["token_idx"] == ex["top_predictions"][0]["token_id"])
            for ex in data["results"]
        ]

        avg_prob = mean(target_probs)
        avg_accuracy = mean(accuracy)
        var_prob = variance(target_probs)

        results.append(
            {
                "dropout_rate": dropout_rate,
                "avg_prob": avg_prob,
                "var_prob": var_prob,
                "avg_accuracy": avg_accuracy,
            }
        )

        lowest_k = sorted(data["results"], key=lambda ex: ex["target"]["token_prob"])[
            :top_k
        ]
        poor_performance_examples[dropout_rate] = lowest_k

    return results, poor_performance_examples


def plot_results(
    results: List[Dict[str, Any]], figures_dir: Path, results_dir: Path
) -> None:
    """
    Generate and save plots from the experiment results.
    """
    # Name figures like this: gpt2_pre2sft_fmfa_fa_s1_dropout_vs_prob.png
    # Load training config from training_config.json located in the same directory as the results
    experiment_config = json.load(open(results_dir / "experiment_config.json"))
    model = experiment_config["model"]
    direction = experiment_config["model_config"]["patch_direction"]
    dataset_name = experiment_config["data_options"]["dataset_name"]
    patch = experiment_config["patch_config_filename"].split(".")[0]
    figure_prefix = f"{model}_{direction}_{dataset_name}_{patch}"

    # Sort results by dropout rate
    results.sort(key=lambda x: x["dropout_rate"])

    # Extract data for plotting
    dropout_rates = [r["dropout_rate"] for r in results]
    avg_probs = [r["avg_prob"] for r in results]
    var_probs = [r["var_prob"] for r in results]
    accuracies = [r["avg_accuracy"] for r in results]

    # Plot 1: Avg Probability with error bars
    plt.figure(figsize=(8, 6))
    plt.errorbar(dropout_rates, avg_probs, yerr=var_probs, fmt="o-", capsize=5)
    plt.xlabel("Dropout Rate")
    plt.ylabel("Average Target Token Probability")
    plt.title(figure_prefix)
    plt.ylim(-0.3, 1.3)
    plt.grid(True)
    LOGGER.info(
        f"Saving figure to {figures_dir / f'{figure_prefix}_dropout_vs_prob.png'}"
    )
    plt.savefig(
        figures_dir / f"{figure_prefix}_dropout_vs_prob.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 2: Accuracy
    plt.figure(figsize=(8, 6))
    plt.errorbar(dropout_rates, accuracies, fmt="o-", capsize=5)
    plt.xlabel("Dropout Rate")
    plt.ylabel("Average Accuracy")
    plt.title(figure_prefix)
    plt.ylim(-0.3, 1.3)
    plt.grid(True)
    LOGGER.info(
        f"Saving figure to {figures_dir / f'{figure_prefix}_dropout_vs_accuracy.png'}"
    )
    plt.savefig(
        figures_dir / f"{figure_prefix}_dropout_vs_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def analyze_performance(
    poor_performance_examples: Dict[float, List[Dict[str, Any]]], figures_dir: Path
) -> None:
    # Create a summary of examples with poor performance
    summary = {}
    for dropout_rate, examples in poor_performance_examples.items():
        example_counts = {}
        for example in examples:
            example_counts[example["target"]["token"]] = (
                example_counts.get(example["target"]["token"], 0) + 1
            )
        summary[dropout_rate] = example_counts

    # Save summary to file
    with open(figures_dir / "performance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def analyze_experiments(cfg) -> None:
    if hasattr(cfg.paths, "results_dir") and cfg.paths.results_dir:
        results_dir = Path(cfg.paths.results_dir)
    else:
        patch_description = cfg.patch_config_filename.split(".")[0]
        if "config_patches_" in patch_description:
            patch_description = patch_description.split("config_patches_")[1]
        results_dir = get_experiment_timestamp_dir(
            cfg.model.pretrained,
            cfg.paths.both_directions_parent,
            cfg.paths.both_directions_checkpoint,
            cfg.model.patch_direction,
            patch_description,
            cfg.paths.dataset_name,
            cfg.timestamp,
            cfg.smoke_test,
        )

    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        LOGGER.info(f"Analyzing results in: {subdir}")
        figures_dir = subdir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            results, poor_performance_examples = load_experiment_results(subdir)
            plot_results(results, figures_dir, results_dir)
            analyze_performance(poor_performance_examples, figures_dir)
        except Exception as e:
            LOGGER.warning(f"Skipping {subdir} due to error: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--timestamp", type=str, default=TIMESTAMP)
    parser.add_argument(
        "--experiment-config", type=str, default="config_experiments.yaml"
    )
    parser.add_argument("--patch-config", type=str, default="no_patching.yaml")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Note: This is hacky and requires the full absolute path",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config entries with KEY=VALUE pairs",
    )
    args = parser.parse_args()

    if not args.experiment_config.endswith(".yaml"):
        args.experiment_config += ".yaml"
    if not args.patch_config.endswith(".yaml"):
        args.patch_config += ".yaml"

    experiment_config_path = EXPERIMENTS_CONFIG_DIR / args.experiment_config
    patch_config_path = PATCH_CONFIG_DIR / args.patch_config

    cfg = load_experiment_config(
        experiment_config_path,
        patch_config_path,
        timestamp=args.timestamp,
        patch_filename=args.patch_config.split("/")[-1],
        overrides=args.override,
    )

    if args.results_dir:
        cfg.paths.results_dir = args.results_dir

    analyze_experiments(cfg)
