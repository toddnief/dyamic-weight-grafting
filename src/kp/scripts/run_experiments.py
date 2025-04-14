import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import torch
import yaml
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


def get_patches(ex, n_layers, test_sentence_template, tokenizer):
    layers_dict = get_layers_dict(n_layers)

    # TODO: Should spaces be managed here or elsewhere?
    # TODO: In general, I don't like how this works
    first_entity = ex["first_actor"]
    movie = " " + ex["movie_title"]
    preposition = " alongside"

    first_entity_tokens = tokenizer.encode(
        first_entity, add_special_tokens=False, return_tensors="pt"
    )
    movie_tokens = tokenizer.encode(
        movie, add_special_tokens=False, return_tensors="pt"
    )
    preposition_tokens = tokenizer.encode(
        preposition, add_special_tokens=False, return_tensors="pt"
    )

    # TODO: This is ugly also
    test_sentence = test_sentence_template.format(**ex, preposition=preposition)
    inputs = tokenizer(test_sentence, return_tensors="pt").to(DEVICE)

    # Set up first entity patch config
    first_entity_start_idx, first_entity_end_idx = find_sublist_index(
        inputs["input_ids"], first_entity_tokens
    )
    first_entity_patch_targets = PatchTargets(
        mlp_up=True, mlp_down=True, o=True, q=False, gate=True
    )

    first_entity_patch_config = {
        "indeces": (first_entity_start_idx, first_entity_end_idx),
        "targets": first_entity_patch_targets,
        "patch_layers": layers_dict["all"],
    }

    # Set up movie patch config
    movie_start_idx, movie_end_idx = find_sublist_index(
        inputs["input_ids"], movie_tokens
    )
    movie_patch_targets = PatchTargets(
        mlp_up=True, mlp_down=True, o=True, q=False, gate=True
    )

    movie_patch_config = {
        "indeces": (movie_start_idx, movie_end_idx),
        "targets": movie_patch_targets,
        "patch_layers": layers_dict["all"],
    }

    # Set up preposition patch config
    preposition_start_idx, preposition_end_idx = find_sublist_index(
        inputs["input_ids"], preposition_tokens
    )
    preposition_patch_targets = PatchTargets(
        mlp_up=True, mlp_down=True, o=True, q=False, gate=True
    )

    preposition_patch_config = {
        "indeces": (preposition_start_idx, preposition_end_idx),
        "targets": preposition_patch_targets,
        "patch_layers": layers_dict["all"],
    }

    patch_configs = [
        first_entity_patch_config,
        movie_patch_config,
        preposition_patch_config,
    ]

    # # TODO: Get a better name...
    # patch_config = {
    #     "first_entity": first_entity_patch_config,
    #     "movie": movie_patch_config,
    #     "preposition": preposition_patch_config,
    # }

    patches = {}
    for patch_config in patch_configs:
        for token_idx in range(patch_config["indeces"][0], patch_config["indeces"][1]):
            patches[token_idx] = Patch(token_idx, **patch_config)

    breakpoint()
    # patches = []
    # for token_idx in range(len(inputs["input_ids"][0])):
    #     if first_entity_start_idx <= token_idx < first_entity_end_idx:
    #         LOGGER.info(f"Patching first entity for token {token_idx}")
    #         patches.append(Patch(token_idx, **first_entity_patch_config))
    #     elif movie_start_idx <= token_idx < movie_end_idx:
    #         LOGGER.info(f"Patching movie for token {token_idx}")
    #         patches.append(Patch(token_idx, **movie_patch_config))
    #     elif preposition_start_idx <= token_idx < preposition_end_idx:
    #         LOGGER.info(f"Patching preposition for token {token_idx}")
    #         patches.append(Patch(token_idx, **preposition_patch_config))
    #     else:
    #         LOGGER.info(f"No patching for token {token_idx}")
    #         patches.append(Patch(token_idx))

    # patches = []
    # for token_idx in range(len(inputs["input_ids"][0])):
    #     patched = False
    #     # Iterate through the configured patches.
    #     for key, (start_idx, end_idx) in phrase_indices.items():
    #         if start_idx <= token_idx < end_idx:
    #             conf = phrases_config[key]
    #             # Decide on patch_layers: if "all", use all_layers; else, use provided list.
    #             patch_layers = all_layers if conf.get("patch_layers") == "all" else conf.get("patch_layers")
    #             # Create patch config dict using the patch_targets from config.
    #             patch_conf = {"targets": PatchTargets(**conf["patch_targets"]), "patch_layers": patch_layers}
    #             LOGGER.info(f"Patching {key} for token {token_idx}")
    #             patches.append(Patch(token_idx, **patch_conf))
    #             patched = True
    #             break
    #     if not patched:
    #         LOGGER.info(f"No patching for token {token_idx}")
    #         patches.append(Patch(token_idx))
    return patches, inputs


def run_patching_experiment(
    ex,
    patches,
    model_config,
    tokenizer,
    inputs,
    llm_recipient_base,
    llm_donor_base,
    target_key,
    patch_dropout=0.0,
    dropout_strategy="layer",  # choices: layer, matrix
    top_k=20,
):
    target_token = ex[target_key]
    target_token_idx = tokenizer.encode(" " + target_token, add_special_tokens=False)[0]

    ex_id = ex["id"]

    past_key_values = None
    for i, p in enumerate(patches):
        # Reset models for new patching
        llm_recipient = copy.deepcopy(llm_recipient_base)
        llm_donor = copy.deepcopy(llm_donor_base)

        LOGGER.info(f"######## PATCH {i + 1} ##########")
        LOGGER.info(
            tokenizer.decode(
                inputs["input_ids"][:, p.patch_token_idx : p.patch_token_idx + 1]
                .squeeze()
                .tolist()
            )
        )
        LOGGER.info(
            f"Patch token start: {p.patch_token_idx}, Patch token end: {p.patch_token_idx}"
        )

        if p.patch_layers is not None:
            LOGGER.info(p.patch_layers)
            for layer_idx in p.patch_layers:
                if dropout_strategy == "layer" and random.random() < patch_dropout:
                    LOGGER.info(f"Skipping layer {layer_idx}")
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
            LOGGER.info("No patching")

        # Get the patched output
        with torch.no_grad():
            patched_output = llm_recipient(
                inputs["input_ids"][:, p.patch_token_idx : p.patch_token_idx + 1],
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = patched_output.past_key_values

    probs = torch.softmax(patched_output.logits[0, -1], dim=-1)
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

    LOGGER.info("##### FINAL patched_output ######")
    LOGGER.info(f"Decoded top token: {top_predictions[0]['token']}")
    LOGGER.info(f"Decoded top prob: {top_predictions[0]['probability']}")
    LOGGER.info(f"Target token: {target_token}")
    LOGGER.info(f"Target token prob: {target_token_prob}")

    results = {
        "ex_id": ex_id,
        "top_predictions": top_predictions,
        "target": {
            "token": target_token,
            "token_idx": target_token_idx,
            "token_prob": target_token_prob,
        },
    }

    return results


def main(config_path):
    # TODO: Clean up these config settings and path variable names
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    SMOKE_TEST = config["smoke_test"]
    model_name = config["model"]["pretrained"]
    experiment_name = config["experiment_name"]

    # Set up dirs
    metadata_path = config["paths"]["metadata"]
    output_dir = EXPERIMENTS_DIR / experiment_name / TIMESTAMP
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    pretrained = MODEL_TO_HFID[model_name]
    finetuned_path = config["paths"]["finetuned"]

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    llm_pretrained = AutoModelForCausalLM.from_pretrained(pretrained).to(DEVICE)
    llm_finetuned = AutoModelForCausalLM.from_pretrained(finetuned_path).to(DEVICE)

    if config["model"]["donor_model"] == "finetuned":
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

    experiment_config = config["experiment_config"]

    # TODO: Need to add this to an experimental config or something
    test_sentence_template = "{first_actor} stars in {movie_title}{preposition}"  # Note: remove spaces for tokenization purposes

    results = []
    limit = 5 if SMOKE_TEST else None
    for ex in metadata[:limit]:
        patches, inputs = get_patches(ex, n_layers, test_sentence_template, tokenizer)
        results.append(
            run_patching_experiment(
                ex,
                patches,
                model_config,
                tokenizer,
                inputs,
                llm_recipient_base=llm_recipient_base,
                llm_donor_base=llm_donor_base,
                **experiment_config,
            )
        )

    results_final = {
        "config": config,
        "results": results,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_final, f, indent=2)


if __name__ == "__main__":
    # Note: Use argparse to allow submission of config file via slurm
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_experiments.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    config_path = EXPERIMENTS_CONFIG_DIR / args.config
    LOGGER.info(f"Running experiments with config: {config_path}")
    main(config_path)
