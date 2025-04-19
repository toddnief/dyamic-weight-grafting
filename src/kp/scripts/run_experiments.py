import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kp.utils.constants import (
    DEVICE,
    EXPERIMENTS_CONFIG_DIR,
    EXPERIMENTS_DIR,
    LOGGER,
    MODEL_TO_HFID,
    PATCH_CONFIG_DIR,
    TIMESTAMP,
)

MODEL_CONFIGS = {
    "gemma": {
        "layers": "model.layers",
        "mlp_up": "mlp.up_proj",
        "mlp_down": "mlp.down_proj",
        "gate": "mlp.gate_proj",
        "q": "self_attn.q_proj",
        "k": "self_attn.k_proj",
        "v": "self_attn.v_proj",
        "o": "self_attn.o_proj",
    },
    "pythia-2.8b": {
        "layers": "gpt_neox.layers",
        "mlp_up": "mlp.dense_h_to_4h",
        "mlp_down": "mlp.dense_4h_to_h",
        "q": "attention.query_key_value",  # fused, so must handle specially
        "o": "attention.dense",
    },
}


@dataclass
class PatchTargets:
    embeddings: bool = False
    lm_head: bool = False
    q: bool = False
    k: bool = False
    v: bool = False
    o: bool = False
    gate: bool = False
    mlp_up: bool = False
    mlp_down: bool = False


@dataclass
class Patch:
    patch_token_idx: int  # TODO: Should I remove this and use indeces instead?
    indeces: Tuple[int, int]
    patch_layers: List[int] = field(default_factory=list)
    targets: PatchTargets = field(default_factory=PatchTargets)


def find_sublist_index(full_list, sublist):
    full_list = full_list.view(-1)
    sublist = sublist.view(-1)
    full_list = full_list.to(DEVICE).tolist()
    sublist = sublist.to(DEVICE).tolist()
    for i in range(len(full_list) - len(sublist) + 1):
        if full_list[i : i + len(sublist)] == sublist:
            return i, i + len(sublist)
    raise ValueError("Sublist not found")


def parse_layers(patch_layers, layers_dict=None):
    """
    Parse layer specifications into a list of layer indices.

    Args:
        patch_layers: Can be None, a string key, a list of items, or a single item
        layers_dict: Dictionary mapping string keys to lists of layer indices

    Returns:
        List of layer indices or None if patch_layers is None
    """
    if patch_layers is None:
        return None

    if isinstance(patch_layers, str):
        # If it's a string, look it up in layers_dict
        if layers_dict is not None and patch_layers in layers_dict:
            return layers_dict[patch_layers]
        else:
            raise ValueError(f"Unknown layer group: {patch_layers}")

    if isinstance(patch_layers, list):
        # If it's a list, process each item
        expanded_layers = []
        for item in patch_layers:
            if (
                isinstance(item, str)
                and layers_dict is not None
                and item in layers_dict
            ):
                # If it's a string key in layers_dict, add those layers
                expanded_layers.extend(layers_dict[item])
            elif isinstance(item, (int, range)):
                # If it's an int or range, add it directly
                if isinstance(item, range):
                    expanded_layers.extend(item)
                else:
                    expanded_layers.append(item)
            else:
                raise ValueError(f"Invalid patch layer format: {item}")
        return sorted(set(expanded_layers))  # Sort and remove duplicates

    if isinstance(patch_layers, (int, range)):
        # Handle single int or range
        if isinstance(patch_layers, range):
            return sorted(set(patch_layers))
        else:
            return [patch_layers]

    raise ValueError(f"Invalid patch layers format: {patch_layers}")


def get_attr(obj, attr_path):
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def patch_component(llm_receipient, llm_donor, base_path, layer_idx, attr_name):
    receipient_layer = get_attr(llm_receipient, f"{base_path}.{layer_idx}")
    donor_layer = get_attr(llm_donor, f"{base_path}.{layer_idx}")
    receipient_component = get_attr(receipient_layer, attr_name)
    donor_component = get_attr(donor_layer, attr_name)
    receipient_component.load_state_dict(donor_component.state_dict())


def get_layers_dict(n_layers):
    all_layers = list(range(n_layers))
    quarter = n_layers // 4
    first_quarter_layers = list(range(0, quarter))
    second_quarter_layers = list(range(quarter, 2 * quarter))
    third_quarter_layers = list(range(2 * quarter, 3 * quarter))
    fourth_quarter_layers = list(range(3 * quarter, n_layers))

    layers_dict = {
        "all": all_layers,
        "first_quarter": first_quarter_layers,
        "second_quarter": second_quarter_layers,
        "third_quarter": third_quarter_layers,
        "fourth_quarter": fourth_quarter_layers,
    }

    return layers_dict


def get_inputs(ex, test_sentence_template, tokenizer):
    # TODO: fix this so preposition is handled correctly
    preposition = " alongside"

    test_sentence = test_sentence_template.format(**ex, preposition=preposition)
    inputs = tokenizer(test_sentence, return_tensors="pt").to(DEVICE)
    return inputs


def get_patches(ex, patch_config, n_layers, tokenizer, input_ids):
    layers_dict = get_layers_dict(n_layers)

    patches = {}
    # Fill all patches with "other" if present
    if "other" in patch_config["patches"]:
        patch_spec = patch_config["patches"]["other"]
        patch_layers = parse_layers(patch_spec.get("layers"), layers_dict)

        for token_idx in range(len(input_ids[0])):
            patches[token_idx] = Patch(
                token_idx,
                indeces=(0, len(input_ids[0])),
                targets=PatchTargets(**patch_spec["targets"]),
                patch_layers=patch_layers,
            )

    # Replace other specified patches
    for key, patch_spec in patch_config["patches"].items():
        if key == "other":
            continue
        # TODO: I hate this and need to refactor to handle the preoposition, etc.
        # Get the span text either from the example or directly
        if "value" in patch_spec:
            span = patch_spec["value"]
        else:
            span = patch_spec.get("prefix", "") + ex[patch_spec["key"]]

        tokens = tokenizer.encode(span, add_special_tokens=False, return_tensors="pt")
        start_idx, end_idx = find_sublist_index(input_ids, tokens)

        targets = PatchTargets(**patch_spec["targets"])
        patch_layers = parse_layers(patch_spec.get("layers"), layers_dict)

        for token_idx in range(start_idx, end_idx):
            patches[token_idx] = Patch(
                token_idx,
                indeces=(start_idx, end_idx),
                targets=targets,
                patch_layers=patch_layers,
            )
    return patches
    # return patches, inputs


def run_patched_inference(
    inputs,
    patches,
    llm_recipient_base,
    llm_donor_base,
    model_config,
    patch_dropout=0.0,
    dropout_strategy="layer",  # choices: layer, matrix
):
    # Initialize cache and models before loop
    kv_cache = None
    llm_recipient = copy.deepcopy(llm_recipient_base)
    llm_donor = copy.deepcopy(llm_donor_base)

    for idx in range(len(inputs["input_ids"][0])):
        dropout = {
            "layers": [],
        }
        if idx in patches and patches[idx].patch_layers:
            p = patches[idx]

            # Reset models for patching
            llm_recipient = copy.deepcopy(llm_recipient_base)
            llm_donor = copy.deepcopy(llm_donor_base)

            for layer_idx in p.patch_layers:
                if dropout_strategy == "layer" and random.random() < patch_dropout:
                    dropout["layers"].append(layer_idx)
                    continue
                for logical_name, physical_name in model_config.items():
                    PATCH_FLAG = (
                        random.random() < patch_dropout
                        if dropout_strategy == "matrix"
                        else True
                    )
                    if PATCH_FLAG and asdict(p.targets).get(logical_name, False):
                        # LOGGER.info(f"Patching {logical_name} at layer {layer_idx}")
                        patch_component(
                            llm_recipient,
                            llm_donor,
                            model_config["layers"],
                            layer_idx,
                            physical_name,
                        )
        else:
            pass

        # Get the patched output
        with torch.no_grad():
            # Try the new cache API first, fall back to past_key_values if needed
            try:
                patched_output = llm_recipient(
                    inputs["input_ids"][:, idx : idx + 1],
                    use_cache=True,
                    cache=kv_cache,
                )
                kv_cache = patched_output.cache
            except TypeError:
                # Fall back to the old API
                patched_output = llm_recipient(
                    inputs["input_ids"][:, idx : idx + 1],
                    use_cache=True,
                    past_key_values=kv_cache,
                )
                kv_cache = patched_output.past_key_values

    probs = torch.softmax(patched_output.logits[0, -1], dim=-1)
    return probs, dropout


def main(experiment_config, patch_config):
    SMOKE_TEST = experiment_config["smoke_test"]
    PATCHING = experiment_config["patching"]
    model_name = experiment_config["model"]["pretrained"]
    dataset_name = experiment_config["dataset_name"]
    patch_direction = experiment_config["model"]["patch_direction"]
    patch_config_filename = experiment_config["patch_config_filename"]
    patch_description = patch_config_filename.split(".")[0]
    patch_description = (
        patch_description.split("config_patches_")[1]
        if "config_patches_" in patch_description
        else patch_description
    )
    timestamp = experiment_config["timestamp"]

    inference_settings = experiment_config["inference_settings"]
    reporting_settings = experiment_config["reporting_settings"]

    # Set up dirs
    metadata_path = experiment_config["paths"]["metadata"]
    timestamp_dir = timestamp + "_smoke_test" if SMOKE_TEST else timestamp
    if PATCHING:
        hyperparams_dir = "dropout_" + str(inference_settings["patch_dropout"])
    else:
        hyperparams_dir = "no_patching"

    EXPERIMENT_NAME = (
        f"{dataset_name}_{model_name}_{patch_direction}_{patch_description}"
    )
    output_dir = EXPERIMENTS_DIR / EXPERIMENT_NAME / timestamp_dir / hyperparams_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    pretrained = MODEL_TO_HFID[model_name]
    both_directions_path = experiment_config["paths"]["both_directions"]
    one_direction_path = experiment_config["paths"]["one_direction"]

    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    if patch_direction == "sft2pre":
        llm_donor_base = AutoModelForCausalLM.from_pretrained(both_directions_path).to(
            DEVICE
        )
        llm_recipient_base = AutoModelForCausalLM.from_pretrained(pretrained).to(DEVICE)

    elif patch_direction == "pre2sft":
        llm_donor_base = AutoModelForCausalLM.from_pretrained(pretrained).to(DEVICE)
        llm_recipient_base = AutoModelForCausalLM.from_pretrained(
            both_directions_path
        ).to(DEVICE)
    elif patch_direction == "both2one":
        llm_donor_base = AutoModelForCausalLM.from_pretrained(both_directions_path).to(
            DEVICE
        )
        llm_recipient_base = AutoModelForCausalLM.from_pretrained(
            one_direction_path
        ).to(DEVICE)

    with open(metadata_path, "r") as f:
        metadata = [json.loads(line) for line in f]

    # Load experiment config
    model_config = MODEL_CONFIGS[model_name]
    n_layers = len(get_attr(llm_recipient_base, model_config["layers"]))

    test_sentence_template = experiment_config["templates"]["test_sentence_template"]

    limit = 5 if SMOKE_TEST else None
    results = []
    for ex in tqdm(metadata[:limit]):
        inputs = get_inputs(
            ex,
            test_sentence_template,
            tokenizer,
        )

        if PATCHING:
            patches = get_patches(
                ex,
                patch_config,
                n_layers,
                tokenizer,
                inputs["input_ids"],
            )
            probs, dropout_record = run_patched_inference(
                inputs,
                patches,
                llm_recipient_base,
                llm_donor_base,
                model_config,
                **inference_settings,
            )
        else:
            dropout_record = {
                "layers": [],
            }
            probs = torch.softmax(
                llm_recipient_base(inputs["input_ids"]).logits[0, -1], dim=-1
            )

        target_name = ex[reporting_settings["target_key"]]
        target_token_idx = tokenizer.encode(
            " " + target_name, add_special_tokens=False
        )[0]
        target_token = tokenizer.decode(target_token_idx)

        topk_probs, topk_indices = torch.topk(probs, reporting_settings["top_k"])
        target_token_prob = probs[target_token_idx].item()

        top_predictions = []
        for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
            token_str = tokenizer.decode([idx])
            top_predictions.append(
                {
                    "token_id": idx,
                    "token": token_str,
                    "probability": prob,
                }
            )

        results.append(
            {
                "ex_id": ex["id"],
                "dropout_record": dropout_record,
                "top_predictions": top_predictions,
                "target": {
                    "token": target_token,
                    "token_idx": target_token_idx,
                    "token_prob": target_token_prob,
                },
            }
        )

    results_with_settings = {
        "inference_settings": inference_settings,
        "patch_config": patch_config,
        "results": results,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_with_settings, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        default="config_experiments.yaml",
        help="Path to the experiment config file",
    )
    parser.add_argument(
        "--patch_config",
        type=str,
        default="config_patches.yaml",
        help="Path to the patch config file",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=TIMESTAMP,
        help="Timestamp for the experiment",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config entries with KEY=VALUE pairs",
    )
    args = parser.parse_args()

    experiment_config_path = EXPERIMENTS_CONFIG_DIR / args.experiment_config
    patch_config_path = PATCH_CONFIG_DIR / args.patch_config
    LOGGER.info(f"Running experiments with experiment config: {experiment_config_path}")
    LOGGER.info(f"Running experiments with patch config: {patch_config_path}")

    with open(experiment_config_path, "r") as f:
        experiment_config = yaml.safe_load(f)
    with open(patch_config_path, "r") as f:
        patch_config = yaml.safe_load(f)

    # Set timestamp passed from command line so experiments scheduled with slurm all have the same timestamp
    experiment_config["timestamp"] = args.timestamp

    # Split the filename from the path
    experiment_config["patch_config_filename"] = args.patch_config.split("/")[-1]

    def set_nested(config, key_path, value):
        keys = key_path.split(".")
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value

    # Usage: --override experiment_config.patch_dropout=0.1
    for item in args.override:
        key, val = item.split("=", 1)
        set_nested(experiment_config, key, yaml.safe_load(val))

    main(experiment_config, patch_config)
