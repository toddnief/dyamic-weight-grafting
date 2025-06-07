import argparse
import copy
import json
import random
import string
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from kg.train.model_factory import model_factory
from kg.utils.constants import (
    DATA_DIR,
    DEVICE,
    EXPERIMENTS_CONFIG_DIR,
    EXPERIMENTS_DIR,
    LOGGER,
    MODEL_TO_HFID,
    PATCH_CONFIG_DIR,
    TIMESTAMP,
    TRAINED_MODELS_DIR,
)
from kg.utils.utils_io import load_experiment_config, namespace_to_dict

MODEL_CONFIGS = {
    "gemma": {
        "layers": "model.layers",
        "components": {
            "ln_1": {"component_path": "input_layernorm"},
            "ln_2": {"component_path": "post_attention_layernorm"},
            "mlp_up": {"component_path": "mlp.up_proj"},
            "mlp_down": {"component_path": "mlp.down_proj"},
            "gate": {"component_path": "mlp.gate_proj"},
            "q": {"component_path": "self_attn.q_proj"},
            "k": {"component_path": "self_attn.k_proj"},
            "v": {"component_path": "self_attn.v_proj"},
            "o": {"component_path": "self_attn.o_proj"},
        },
        "embeddings": {"component_path": "model.embed_tokens"},
        "ln_f": {"component_path": "model.norm"},
        "lm_head": {"component_path": "lm_head"},
    },
    "pythia-2.8b": {
        "layers": "gpt_neox.layers",
        "components": {
            "ln_1": {"component_path": "input_layernorm"},
            "ln_2": {"component_path": "post_attention_layernorm"},
            "mlp_up": {"component_path": "mlp.dense_h_to_4h"},
            "mlp_down": {"component_path": "mlp.dense_4h_to_h"},
            "q": {"component_path": "attention.query_key_value"},
            # "q": {
            #     "component_path": "attention.query_key_value",
            #     "slice_range": slice(0, 768),
            # },
            # "k": {
            #     "component_path": "attention.query_key_value",
            #     "slice_range": slice(768, 1536),
            # },
            # "v": {
            #     "component_path": "attention.query_key_value",
            #     "slice_range": slice(1536, 2304),
            # },
            "o": {"component_path": "attention.dense"},
        },
        "embeddings": {"component_path": "gpt_neox.embed_in"},
        "ln_f": {"component_path": "gpt_neox.final_layer_norm"},
        "lm_head": {"component_path": "embed_out"},
    },
    "gpt2": {
        "layers": "transformer.h",
        "components": {
            "mlp_up": {"component_path": "mlp.c_fc"},
            "mlp_down": {"component_path": "mlp.c_proj"},
            "q": {"component_path": "attn.c_attn"},
            # "q": {"component_path": "attn.c_attn", "slice_range": slice(0, 768)},
            # "k": {"component_path": "attn.c_attn", "slice_range": slice(768, 1536)},
            # "v": {"component_path": "attn.c_attn", "slice_range": slice(1536, 2304)},
            "o": {"component_path": "attn.c_proj"},
        },
    },
    "gpt2-xl": {
        "layers": "transformer.h",
        "components": {
            "ln_1": {"component_path": "ln_1"},
            "ln_2": {"component_path": "ln_2"},
            "mlp_up": {"component_path": "mlp.c_fc"},
            "mlp_down": {"component_path": "mlp.c_proj"},
            "q": {"component_path": "attn.c_attn"},
            # "q": {"component_path": "attn.c_attn", "slice_range": slice(0, 1600)},
            # "k": {"component_path": "attn.c_attn", "slice_range": slice(1600, 3200)},
            # "v": {"component_path": "attn.c_attn", "slice_range": slice(3200, 4800)},
            "o": {"component_path": "attn.c_proj"},
        },
        "embeddings": {"component_path": "transformer.wte"},
        "ln_f": {"component_path": "transformer.ln_f"},
        "lm_head": {"component_path": "lm_head"},
    },
    "llama3": {
        "layers": "model.layers",
        "components": {
            "ln_1": {"component_path": "input_layernorm"},
            "ln_2": {"component_path": "post_attention_layernorm"},
            "mlp_up": {"component_path": "mlp.up_proj"},
            "mlp_down": {"component_path": "mlp.down_proj"},
            "gate": {"component_path": "mlp.gate_proj"},
            "q": {"component_path": "self_attn.q_proj"},
            "k": {"component_path": "self_attn.k_proj"},
            "v": {"component_path": "self_attn.v_proj"},
            "o": {"component_path": "self_attn.o_proj"},
        },
        "embeddings": {"component_path": "model.embed_tokens"},
        "ln_f": {"component_path": "model.norm"},
        "lm_head": {"component_path": "lm_head"},
    },
    "olmo": {
        "layers": "model.transformer.blocks",
        "components": {
            "ln_1": {"component_path": "attn_norm"},
            "ln_2": {"component_path": "ff_norm"},
            "mlp_up": {"component_path": "ff_proj"},
            "mlp_down": {"component_path": "ff_out"},
            "q": {"component_path": "att_proj"},
            # "q": {"component_path": "att_proj", "slice_range": [0, 2048]},
            # "k": {"component_path": "att_proj", "slice_range": [2048, 4096]},
            # "v": {"component_path": "att_proj", "slice_range": [4096, 6144]},
            "o": {"component_path": "attn_out"},
        },
        "embeddings": {"component_path": "model.transformer.wte"},
        "ln_f": {"component_path": "model.ln_f"},
        "lm_head": {"component_path": "lm_head"},
    },
}


@dataclass
class PatchTargets:
    embeddings: bool = False
    q: bool = False
    k: bool = False
    v: bool = False
    o: bool = False
    gate: bool = False
    mlp_up: bool = False
    mlp_down: bool = False
    ln_1: bool = False
    ln_2: bool = False


@dataclass
class Patch:
    patch_token_idx: int
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
        if attr.isdigit():
            obj = obj[int(attr)]
        else:
            obj = getattr(obj, attr)
    return obj


# TODO: Add SVD patching (need to cache SVD)
def patch_component(
    llm_donor,
    llm_recipient,
    layers_path,
    layer_idx,
    component_path,
    slice_range: Optional[slice] = None,
    log_patches=False,
):
    recipient_layer = get_attr(llm_recipient, f"{layers_path}.{layer_idx}")
    recipient_component = get_attr(recipient_layer, component_path)

    donor_layer = get_attr(llm_donor, f"{layers_path}.{layer_idx}")
    donor_component = get_attr(donor_layer, component_path)

    if slice_range is None:
        recipient_component.load_state_dict(donor_component.state_dict())
    else:
        # Note: for some models, we should slice along the columns, for others, along the rows
        slice_axis = (
            1 if donor_component.weight.size(1) > donor_component.weight.size(0) else 0
        )
        with torch.no_grad():
            if slice_axis == 1:
                # Column slice
                if slice_range.stop > donor_component.weight.size(
                    1
                ) or slice_range.stop > recipient_component.weight.size(1):
                    raise IndexError(
                        f"Slice {slice_range} is out of range for column slicing"
                    )
                if log_patches:
                    LOGGER.info(f"Patching columns: {slice_range}")
                recipient_component.weight[:, slice_range] = donor_component.weight[
                    :, slice_range
                ]
            else:
                # Row slice (if GPT2-small or similar model)
                if slice_range.stop > donor_component.weight.size(
                    0
                ) or slice_range.stop > recipient_component.weight.size(0):
                    raise IndexError(
                        f"Slice {slice_range} is out of range for row slicing"
                    )
                if log_patches:
                    LOGGER.info(f"Patching rows: {slice_range}")
                recipient_component.weight[slice_range, :] = donor_component.weight[
                    slice_range, :
                ]

            # Note: Not all models have biases
            if (
                hasattr(donor_component, "bias")
                and hasattr(recipient_component, "bias")
                and donor_component.bias is not None
                and recipient_component.bias is not None
            ):
                bias_size = recipient_component.bias.size(0)
                if slice_axis == 1:
                    # Column slicing (bias is 1D, we must index directly)
                    slice_range = slice(
                        slice_range.start, min(slice_range.stop, bias_size)
                    )
                if log_patches:
                    LOGGER.info(f"Patching bias: {slice_range}")
                recipient_component.bias[slice_range] = donor_component.bias[
                    slice_range
                ]


def get_layers_dict(n_layers):
    all_layers = list(range(n_layers))
    quarter = n_layers // 4
    first_quarter_layers = list(range(0, quarter))
    second_quarter_layers = list(range(quarter, 2 * quarter))
    third_quarter_layers = list(range(2 * quarter, 3 * quarter))
    fourth_quarter_layers = list(range(3 * quarter, n_layers))
    first_eighth_layers = list(range(0, quarter // 2))
    second_eighth_layers = list(range(quarter // 2, quarter))
    third_eighth_layers = list(range(2 * quarter, 3 * quarter // 2))
    fourth_eighth_layers = list(range(3 * quarter // 2, 2 * quarter))
    fifth_eighth_layers = list(range(2 * quarter, 5 * quarter // 2))
    sixth_eighth_layers = list(range(5 * quarter // 2, 3 * quarter))
    seventh_eighth_layers = list(range(3 * quarter // 2, 7 * quarter // 2))
    eighth_eighth_layers = list(range(7 * quarter // 2, n_layers))

    layers_dict = {
        "all": all_layers,
        "first_quarter": first_quarter_layers,
        "second_quarter": second_quarter_layers,
        "third_quarter": third_quarter_layers,
        "fourth_quarter": fourth_quarter_layers,
        "first_eighth": first_eighth_layers,
        "second_eighth": second_eighth_layers,
        "third_eighth": third_eighth_layers,
        "fourth_eighth": fourth_eighth_layers,
        "fifth_eighth": fifth_eighth_layers,
        "sixth_eighth": sixth_eighth_layers,
    }

    return layers_dict


def get_inputs(ex, test_sentence_template, tokenizer):
    test_sentence = test_sentence_template.format(**ex)
    inputs = tokenizer(test_sentence, return_tensors="pt").to(DEVICE)
    return inputs


def get_patches(
    ex,
    patch_config,
    n_layers,
    tokenizer,
    input_ids,
    test_sentence_template=None,
    # override_layers=False,
):
    if test_sentence_template is not None:
        formatter = string.Formatter()
        test_sentence_fields = [
            fname for _, fname, _, _ in formatter.parse(test_sentence_template) if fname
        ]
        test_sentence_fields = set(test_sentence_fields)
    # TODO: Hacky override for counterfact
    else:
        test_sentence_fields = {"subject"}

    layers_dict = get_layers_dict(n_layers)
    patches = {}

    # Fill all tokens with "other" patch spec if defined
    if hasattr(patch_config.patches, "other"):
        patch_spec = patch_config.patches.other
        patch_layers = parse_layers(getattr(patch_spec, "layers", None), layers_dict)

        for token_idx in range(len(input_ids[0])):
            patches[token_idx] = Patch(
                token_idx,
                targets=PatchTargets(**vars(patch_spec.targets)),
                patch_layers=patch_layers,
            )

    for patch_name, patch_spec in vars(patch_config.patches).items():
        # Skip other patch spec — already handled above
        if (
            patch_name == "other"
            or patch_name == "token_idx"
            or patch_name not in test_sentence_fields
        ):
            continue

        # Otherwise, try to locate span in input_ids (with and without space)
        span = ex[getattr(patch_spec, "key")]
        variants = [span, span.lstrip()] if span.startswith(" ") else [span, " " + span]

        for variant in variants:
            token_ids = tokenizer.encode(
                variant, add_special_tokens=False, return_tensors="pt"
            )
            try:
                start_idx, end_idx = find_sublist_index(input_ids, token_ids)
                break
            except ValueError:
                continue
        else:
            LOGGER.warning(f"Span not found in input_ids for any variant: {variants}")
            return None

        # Extract matrices and layers to patch
        targets = PatchTargets(**vars(patch_spec.targets))

        layers_spec = getattr(patch_spec, "layers", None)

        patch_layers = parse_layers(layers_spec, layers_dict)

        # Add patches for each token in span
        for token_idx in range(start_idx, end_idx):
            patches[token_idx] = Patch(
                token_idx,
                targets=targets,
                patch_layers=patch_layers,
            )

    # Hacky override for index-based patching
    for patch_name, patch_spec in vars(patch_config.patches).items():
        if "token_idx" not in patch_name:
            continue

        layers_spec = getattr(patch_spec, "layers", None)

        for token_idx in patch_spec.values:
            token_idx = int(token_idx)

            if token_idx < 0:  # Handle negative indices
                token_idx = len(input_ids[0]) + token_idx

            if token_idx >= len(input_ids[0]) or token_idx < 0:
                raise IndexError(
                    f"Token index {token_idx} is out of range for input sequence."
                )

            patches[token_idx] = Patch(
                token_idx,
                targets=PatchTargets(**vars(patch_spec.targets)),
                patch_layers=parse_layers(layers_spec, layers_dict),
            )

    return patches


def run_patched_inference(
    inputs,
    patches,
    llm_donor,
    llm_recipient_base,
    model_config,
    tokenizer,
    patch_lm_head="never",  # choices: never, always, last_token
    dropout_rate=0.0,
    dropout_unit="layer",  # choices: layer, matrix
    dropout_strategy="count",  # choices: count, random
    log_patches=False,
    **kwargs,
):
    # Initialize cache and models before loop
    kv_cache = None
    llm_recipient = copy.deepcopy(llm_recipient_base)
    dropout_layers = None

    if patch_lm_head == "never" or patch_lm_head == "last_token":
        patch_lm_head = False
    elif patch_lm_head == "always":
        patch_lm_head = True
    else:
        raise ValueError(f"Invalid patch_lm_head value: {patch_lm_head}")

    if log_patches:
        LOGGER.info(f"inputs: {inputs}")

    for idx in range(len(inputs["input_ids"][0])):
        # Reset model for patching
        llm_recipient = copy.deepcopy(llm_recipient_base)

        # Note: patches are saved in a dictionary with token indices as keys
        if idx in patches and patches[idx].patch_layers:
            p = patches[idx]
            if log_patches:
                LOGGER.info(
                    f"Patching {p.targets} at layer {p.patch_layers} for token idx {idx}: {tokenizer.decode(inputs['input_ids'][0][idx])}"
                )
            if p.targets.embeddings:
                recipient_embeddings = get_attr(
                    llm_recipient, model_config["embeddings"]["component_path"]
                )
                donor_embeddings = get_attr(
                    llm_donor, model_config["embeddings"]["component_path"]
                )
                recipient_embeddings.load_state_dict(donor_embeddings.state_dict())

            # Determine which layers to drop
            if dropout_strategy == "count":
                dropout_count = int(dropout_rate * len(p.patch_layers))
                dropout_layers = random.sample(p.patch_layers, dropout_count)
            elif dropout_strategy == "random":
                dropout_layers = [
                    layer_idx
                    for layer_idx in p.patch_layers
                    if random.random() < dropout_rate
                ]
            dropout_layers = sorted(set(dropout_layers))

            for layer_idx in p.patch_layers:
                if layer_idx in dropout_layers and dropout_unit == "layer":
                    continue
                for logical_name, component_config in model_config[
                    "components"
                ].items():
                    if asdict(p.targets).get(logical_name, False):
                        # TODO: Add matrix dropout strategy
                        if log_patches:
                            LOGGER.info(
                                f"Patching {logical_name} at layer {layer_idx} for token idx {idx}"
                            )
                        patch_component(
                            llm_donor,
                            llm_recipient,
                            model_config["layers"],
                            layer_idx,
                            log_patches=log_patches,
                            **component_config,
                        )

            # If last index is patched, set patch_lm_head to True
            if idx == len(inputs["input_ids"][0]) - 1 and patch_lm_head == "last_token":
                patch_lm_head = True
        elif log_patches:
            LOGGER.info(f"No patch at token idx {idx}")

        # Get the model output and save kv cache
        with torch.no_grad():
            # Try the new cache API first, fall back to past_key_values if needed
            try:
                patched_output = llm_recipient(
                    inputs["input_ids"][:, idx : idx + 1],
                    use_cache=True,
                    cache=kv_cache,
                    output_hidden_states=True,
                )
                kv_cache = patched_output.cache
            except TypeError:
                # Fall back to the old API
                patched_output = llm_recipient(
                    inputs["input_ids"][:, idx : idx + 1],
                    use_cache=True,
                    past_key_values=kv_cache,
                    output_hidden_states=True,
                )
                kv_cache = patched_output.past_key_values

    if patch_lm_head:
        final_ln = get_attr(llm_donor, model_config["ln_f"]["component_path"])
        lm_head = get_attr(llm_donor, model_config["lm_head"]["component_path"])
        logits = lm_head(final_ln(patched_output.hidden_states[-1]))
    else:
        logits = patched_output.logits

    probs = torch.softmax(logits[0, -1], dim=-1)
    # TODO: dropout layers is only the last token, fix this if we actually need it
    return probs, {"layers": dropout_layers}


def get_experiment_timestamp_dir(
    model_name,
    both_directions_parent,
    both_directions_checkpoint,
    patch_direction,
    patch_description,
    dataset_name,
    timestamp,
    patch_lm_head,
    smoke_test,
    base_experiments_dir=EXPERIMENTS_DIR,
):
    if both_directions_checkpoint is None:
        both_directions_checkpoint = "best_saved_checkpoint"
    if "/" in both_directions_parent:
        both_directions_parent, both_directions_checkpoint = (
            both_directions_parent.split("/")
        )

    # Note: Collapse slashes in parent directory name for consistent experiment dir structure
    both_directions_parent = both_directions_parent.replace("/", "_")
    checkpoint_name = (
        f"{both_directions_parent}_{both_directions_checkpoint}_{timestamp}"
    )

    patch_lm_head = "lm_head" + "_" + patch_lm_head
    # patch_lm_head = f"{patch_lm_head}_smoke_test" if smoke_test else patch_lm_head

    return (
        base_experiments_dir
        / patch_lm_head
        / dataset_name
        / model_name
        / patch_direction
        / patch_description
        / checkpoint_name
    )


def main(cfg):
    smoke_test = bool(cfg.smoke_test)

    models_dir = (
        TRAINED_MODELS_DIR
        / MODEL_TO_HFID[cfg.model.pretrained]
        / cfg.paths.dataset_name
    )
    pretrained_model_name = cfg.model.pretrained

    patch_lm_head = cfg.inference_config.patch_lm_head

    # Load best saved checkpoint if not specified
    if cfg.paths.both_directions_checkpoint is not None:
        both_directions_path = (
            models_dir
            / cfg.paths.both_directions_parent
            / cfg.paths.both_directions_checkpoint
        )
    else:
        both_directions_path = models_dir / cfg.paths.both_directions_parent

    # TODO: This is also an ugly hack to deal with counterfact experiments
    if hasattr(cfg.paths, "one_direction_parent"):
        checkpoint = getattr(cfg.paths, "one_direction_checkpoint", None)
        if checkpoint is not None:
            one_direction_path = (
                models_dir / cfg.paths.one_direction_parent / checkpoint
            )
        elif cfg.paths.one_direction_parent is not None:
            one_direction_path = models_dir / cfg.paths.one_direction_parent
        else:
            one_direction_path = None

    # Derive patch description from filename
    patch_config_filename = cfg.patch_config_filename
    patch_description = patch_config_filename.split(".")[0]
    if "config_patches_" in patch_description:
        patch_description = patch_description.split("config_patches_")[1]

    # Set up directories
    if (
        hasattr(cfg.paths, "experiments_dir_addendum")
        and cfg.paths.experiments_dir_addendum
    ):
        base_experiments_dir = EXPERIMENTS_DIR / cfg.paths.experiments_dir_addendum
    else:
        base_experiments_dir = EXPERIMENTS_DIR

    experiment_timestamp_dir = get_experiment_timestamp_dir(
        pretrained_model_name,
        cfg.paths.both_directions_parent,
        cfg.paths.both_directions_checkpoint,
        cfg.model.patch_direction,
        patch_description,
        cfg.paths.dataset_name,
        cfg.timestamp,
        patch_lm_head,
        smoke_test,
        base_experiments_dir=base_experiments_dir,
    )
    experiment_timestamp_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        f"Saving experiment config to {experiment_timestamp_dir / 'experiment_config.json'}"
    )
    with open(experiment_timestamp_dir / "experiment_config.json", "w") as f:
        json.dump(namespace_to_dict(cfg), f, indent=2)

    if cfg.patching_flag:
        hyperparams_dir = f"dropout_{cfg.inference_config.dropout_rate}_{cfg.inference_config.dropout_unit}_{cfg.inference_config.dropout_strategy}"
    else:
        hyperparams_dir = "no_patching"

    if cfg.model.patch_direction == "sft2pre":
        pretrained = MODEL_TO_HFID[cfg.model.pretrained]
        llm_sft, _, _ = model_factory(str(both_directions_path))
        llm_pretrained, tokenizer, _ = model_factory(pretrained)

        LOGGER.info(f"Loading donor model from {both_directions_path}")
        llm_donor_base = llm_sft
        LOGGER.info(f"Loading recipient model from {pretrained}")
        llm_recipient_base = llm_pretrained
    elif cfg.model.patch_direction == "pre2sft":
        pretrained = MODEL_TO_HFID[cfg.model.pretrained]
        llm_sft, _, _ = model_factory(str(both_directions_path))
        llm_pretrained, tokenizer, _ = model_factory(pretrained)

        LOGGER.info(f"Loading donor model from {pretrained}")
        llm_donor_base = llm_pretrained
        LOGGER.info(f"Loading recipient model from {both_directions_path}")
        llm_recipient_base = llm_sft
    elif cfg.model.patch_direction == "both2one":
        LOGGER.info(f"Loading donor model from {both_directions_path}")
        llm_donor_base, tokenizer, _ = model_factory(str(both_directions_path))
        LOGGER.info(f"Loading recipient model from {one_direction_path}")
        llm_recipient_base, _, _ = model_factory(str(one_direction_path))

    # TODO: Ugly hack to run counterfact experiments
    if cfg.paths.dataset_name == "counterfact":
        dataset_cf = load_dataset("NeelNanda/counterfact-tracing")
    else:
        metadata_path = (
            DATA_DIR
            / cfg.paths.dataset_name
            / cfg.paths.dataset_dir
            / "metadata"
            / "metadata.jsonl"
        )
        with open(metadata_path, "r") as f:
            metadata = [json.loads(line) for line in f]

    model_config = MODEL_CONFIGS[cfg.model.pretrained]
    n_layers = len(get_attr(llm_recipient_base, model_config["layers"]))
    limit = 30 if smoke_test else None

    movie_patches = set(
        [
            "m.yaml",
            "fe_m.yaml",
            "fe_m_lt.yaml",
            "m_lt.yaml",
            "not_fe_m.yaml",
            "not_fe_m_lt.yaml",
            "fe_m_p_lt.yaml",
            "fe_m_p.yaml",
        ]
    )

    # TODO: Ugly hack to run counterfact experiments
    if cfg.paths.dataset_name == "counterfact":
        output_dir = experiment_timestamp_dir / hyperparams_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        log_patches = True
        results = []
        for ex in tqdm(
            dataset_cf["train"]
            if limit is None
            else dataset_cf["train"].select(range(limit))
        ):
            inputs = tokenizer(ex["prompt"], return_tensors="pt").to(DEVICE)

            if cfg.patching_flag:
                # Need to get patches for counterfact...use "subject" for "first entity"
                patches = get_patches(
                    ex, cfg.patch_config, n_layers, tokenizer, inputs["input_ids"]
                )
                if patches is None:
                    continue

                probs, dropout_record = run_patched_inference(
                    inputs,
                    patches,
                    llm_donor_base,
                    llm_recipient_base,
                    model_config,
                    tokenizer,
                    **vars(cfg.inference_config),
                    log_patches=log_patches,
                )
                log_patches = False
            else:
                dropout_record = {"layers": []}
                probs = torch.softmax(
                    llm_recipient_base(inputs["input_ids"]).logits[0, -1], dim=-1
                )

            target_name = ex["subject"]
            target_token_idx = tokenizer.encode(
                " " + target_name, add_special_tokens=False
            )[0]
            target_token = tokenizer.decode(target_token_idx)

            topk_probs, topk_indices = torch.topk(probs, cfg.analysis_config.top_k)
            target_token_prob = probs[target_token_idx].item()

            top_predictions = [
                {
                    "token_id": idx,
                    "token": tokenizer.decode([idx]),
                    "probability": prob,
                }
                for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist())
            ]

            target_token_rank = (probs > target_token_prob).sum().item()

            results.append(
                {
                    "ex_id": ex["relation_id"],
                    "dropout_record": dropout_record,
                    "top_predictions": top_predictions,
                    "target": {
                        "token": target_token,
                        "token_idx": target_token_idx,
                        "token_prob": target_token_prob,
                        "token_rank": target_token_rank,
                    },
                }
            )

        results_with_configs = {
            "inference_config": namespace_to_dict(cfg.inference_config),
            "patch_config": namespace_to_dict(cfg.patch_config),
            "results": results,
        }

        LOGGER.info(f"Saving results to {output_dir / 'results.json'}")
        with open(output_dir / "results.json", "w") as f:
            json.dump(results_with_configs, f, indent=2)

    else:
        for template_name, test_template in vars(cfg.test_templates).items():
            # TODO: hack to only run movie patches for sentence_3
            if (
                template_name != "sentence_3"
                and cfg.patch_config_filename in movie_patches
            ):
                continue

            test_sentence_template = test_template.test_sentence_template
            test_preposition = test_template.preposition
            test_relation = test_template.relation
            test_relation_preposition = test_template.relation_preposition

            # Create nested directory for results
            output_dir = experiment_timestamp_dir / template_name / hyperparams_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            log_patches = True

            results = []
            for ex in tqdm(metadata[:limit]):
                ex["preposition"] = test_preposition
                ex["relation"] = test_relation
                ex["relation_preposition"] = test_relation_preposition
                inputs = get_inputs(ex, test_sentence_template, tokenizer)

                if cfg.patching_flag:
                    patches = get_patches(
                        ex,
                        cfg.patch_config,
                        n_layers,
                        tokenizer,
                        inputs["input_ids"],
                        test_sentence_template,
                    )
                    if patches is None:
                        continue
                    probs, dropout_record = run_patched_inference(
                        inputs,
                        patches,
                        llm_donor_base,
                        llm_recipient_base,
                        model_config,
                        tokenizer,
                        **vars(cfg.inference_config),
                        log_patches=log_patches,
                    )
                    log_patches = False
                else:
                    dropout_record = {"layers": []}
                    probs = torch.softmax(
                        llm_recipient_base(inputs["input_ids"]).logits[0, -1], dim=-1
                    )

                target_name = ex[cfg.analysis_config.target_key]
                target_token_idx = tokenizer.encode(
                    " " + target_name, add_special_tokens=False
                )[0]
                target_token = tokenizer.decode(target_token_idx)

                topk_probs, topk_indices = torch.topk(probs, cfg.analysis_config.top_k)
                target_token_prob = probs[target_token_idx].item()

                top_predictions = [
                    {
                        "token_id": idx,
                        "token": tokenizer.decode([idx]),
                        "probability": prob,
                    }
                    for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist())
                ]

                target_token_rank = (probs > target_token_prob).sum().item()

                results.append(
                    {
                        "ex_id": ex["id"],
                        "dropout_record": dropout_record,
                        "top_predictions": top_predictions,
                        "target": {
                            "token": target_token,
                            "token_idx": target_token_idx,
                            "token_prob": target_token_prob,
                            "token_rank": target_token_rank,
                        },
                    }
                )

            results_with_configs = {
                "inference_config": namespace_to_dict(cfg.inference_config),
                "patch_config": namespace_to_dict(cfg.patch_config),
                "results": results,
            }

            LOGGER.info(f"Saving results to {output_dir / 'results.json'}")
            with open(output_dir / "results.json", "w") as f:
                json.dump(results_with_configs, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        type=str,
        default="config_experiments.yaml",
        help="Path to the experiment config file",
    )
    parser.add_argument(
        "--patch-config",
        type=str,
        default="fe_lt.yaml",
        help="Path to the patch config file (optional if included in experiment config)",
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

    if not args.experiment_config.endswith(".yaml"):
        args.experiment_config += ".yaml"
    if args.patch_config and not args.patch_config.endswith(".yaml"):
        args.patch_config += ".yaml"

    experiment_config_path = EXPERIMENTS_CONFIG_DIR / args.experiment_config
    patch_config_path = (
        PATCH_CONFIG_DIR / args.patch_config if args.patch_config else None
    )
    LOGGER.info(f"Running experiments with experiment config: {experiment_config_path}")
    if patch_config_path:
        LOGGER.info(f"Running experiments with patch config: {patch_config_path}")

    cfg = load_experiment_config(
        experiment_config_path,
        patch_config_path,
        timestamp=args.timestamp,
        patch_filename=args.patch_config.split("/")[-1] if args.patch_config else None,
        overrides=args.override,
    )

    LOGGER.info(f"Running experiment with config: {cfg}")

    main(cfg)
