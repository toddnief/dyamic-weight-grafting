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


def parse_layers(patch_layers):
    expanded_layers = []
    for item in patch_layers:
        if isinstance(item, range):
            expanded_layers.extend(item)
        elif isinstance(item, int):
            expanded_layers.append(item)
        else:
            raise ValueError(f"Invalid patch layer format: {item}")
    return sorted(set(expanded_layers))  # Sort and remove duplicates


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


def get_patches(ex, patch_config, n_layers, test_sentence_template, tokenizer):
    layers_dict = get_layers_dict(n_layers)

    # TODO: In general, I don't like how this works
    # TODO: Should spaces be managed here or elsewhere?
    first_entity = ex["first_actor"]
    movie = " " + ex["movie_title"]
    preposition = " alongside"

    test_sentence = test_sentence_template.format(**ex, preposition=preposition)
    inputs = tokenizer(test_sentence, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    patches = {}
    # Fill all patches with "other" if present
    if "other" in patch_config["patches"]:
        patch_spec = patch_config["patches"]["other"]
        patch_layers = (
            layers_dict[patch_spec["layers"]] if patch_spec["layers"] else None
        )
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
        patch_layers = layers_dict[patch_spec["layers"]]

        for token_idx in range(start_idx, end_idx):
            patches[token_idx] = Patch(
                token_idx,
                indeces=(start_idx, end_idx),
                targets=targets,
                patch_layers=patch_layers,
            )
    return patches, inputs


def run_patched_inference(
    llm_recipient_base,
    llm_donor_base,
    model_config,
    inputs,
    patches,
    patch_dropout=0.0,
    dropout_strategy="layer",  # choices: layer, matrix
):
    # Initialize past key values and models before loop
    past_key_values = None
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
                        # LOGGER.info(f"Patching {logical_name} for layer {layer_idx}")
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
            patched_output = llm_recipient(
                inputs["input_ids"][:, idx : idx + 1],
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = patched_output.past_key_values

    probs = torch.softmax(patched_output.logits[0, -1], dim=-1)
    return probs, dropout


def run_patching_experiment(
    ex,
    patches,
    model_config,
    tokenizer,
    inputs,
    llm_recipient_base,
    llm_donor_base,
    patching=True,
    target_key="second_actor",
    patch_dropout=0.0,
    dropout_strategy="layer",  # choices: layer, matrix
    top_k=20,
):
    target_name = ex[target_key]
    target_token_idx = tokenizer.encode(" " + target_name, add_special_tokens=False)[0]
    target_token = tokenizer.decode(target_token_idx)

    ex_id = ex["id"]

    # TODO: Figure out where to pass the configs for patched inference, etc.
    if patching:
        probs, dropout = run_patched_inference(
            llm_recipient_base,
            llm_donor_base,
            model_config,
            inputs,
            patches,
            patch_dropout,
            dropout_strategy,
        )
    else:
        dropout = {
            "layers": [],
        }
        probs = torch.softmax(
            llm_recipient_base(inputs["input_ids"]).logits[0, -1], dim=-1
        )

    topk_probs, topk_indices = torch.topk(probs, top_k)
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

    results = {
        "ex_id": ex_id,
        "dropout": dropout,
        "top_predictions": top_predictions,
        "target": {
            "token": target_token,
            "token_idx": target_token_idx,
            "token_prob": target_token_prob,
        },
    }

    return results


def main(experiment_config, patch_config):
    SMOKE_TEST = experiment_config["smoke_test"]
    model_name = experiment_config["model"]["pretrained"]
    experiment_name = experiment_config["experiment_name"]
    timestamp = experiment_config["timestamp"]

    # Set up dirs
    metadata_path = experiment_config["paths"]["metadata"]
    timestamp_dir = timestamp + "_smoke_test" if SMOKE_TEST else timestamp
    if experiment_config["experiment_settings"]["patching"]:
        hyperparams_dir = "dropout_" + str(
            experiment_config["experiment_settings"]["patch_dropout"]
        )
    else:
        hyperparams_dir = "no_patching"

    output_dir = EXPERIMENTS_DIR / experiment_name / timestamp_dir / hyperparams_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    pretrained = MODEL_TO_HFID[model_name]
    finetuned_path = experiment_config["paths"]["finetuned"]

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    llm_pretrained = AutoModelForCausalLM.from_pretrained(pretrained).to(DEVICE)
    llm_finetuned = AutoModelForCausalLM.from_pretrained(finetuned_path).to(DEVICE)

    if experiment_config["model"]["donor_model"] == "finetuned":
        llm_donor_base = llm_finetuned
        llm_recipient_base = llm_pretrained
    else:
        llm_donor_base = llm_pretrained
        llm_recipient_base = llm_finetuned

    with open(metadata_path, "r") as f:
        metadata = [json.loads(line) for line in f]

    # Load experiment config
    model_config = MODEL_CONFIGS[model_name]
    n_layers = len(get_attr(llm_pretrained, model_config["layers"]))

    experiment_settings = experiment_config["experiment_settings"]

    test_sentence_template = experiment_config["templates"]["test_sentence_template"]

    results = []
    limit = 5 if SMOKE_TEST else None
    for ex in tqdm(metadata[:limit]):
        patches, inputs = get_patches(
            ex, patch_config, n_layers, test_sentence_template, tokenizer
        )
        results.append(
            run_patching_experiment(
                ex,
                patches,
                model_config,
                tokenizer,
                inputs,
                llm_recipient_base=llm_recipient_base,
                llm_donor_base=llm_donor_base,
                **experiment_settings,
            )
        )

    results_final = {
        "experiment_settings": experiment_settings,
        "patch_config": patch_config,
        "results": results,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_final, f, indent=2)


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
    patch_config_path = EXPERIMENTS_CONFIG_DIR / args.patch_config
    LOGGER.info(f"Running experiments with experiment config: {experiment_config_path}")
    LOGGER.info(f"Running experiments with patch config: {patch_config_path}")

    with open(experiment_config_path, "r") as f:
        experiment_config = yaml.safe_load(f)
    with open(patch_config_path, "r") as f:
        patch_config = yaml.safe_load(f)

    # Set timestamp passed from command line so experiments scheduled with slurm all have the same timestamp
    experiment_config["timestamp"] = args.timestamp

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
