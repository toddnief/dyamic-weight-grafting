import copy
import json
import random
from dataclasses import asdict, dataclass, field
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

pretrained = "google/gemma-1.1-2b-it"
pretrained = "EleutherAI/pythia-2.8b"

llm_pretrained = AutoModelForCausalLM.from_pretrained(pretrained).to(DEVICE)

one_direction = (
    "/net/projects/clab/tnief/bidirectional-reversal/trained/gemma_one_direction"
)
both_directions = (
    "/net/projects/clab/tnief/bidirectional-reversal/trained/gemma_both_directions"
)

both_directions = "/net/projects/clab/tnief/bidirectional-reversal/results/pythia-2.8b/fake_movies_real_actors20250408_1954/checkpoint-7200"
both_directions

path = "/home/tnief/1-Projects/bidirectional-reversal/data/fake_movies_real_actors_2025-04-08_19-50-18/metadata/metadata.jsonl"

tokenizer = AutoTokenizer.from_pretrained(pretrained)

llm_both = AutoModelForCausalLM.from_pretrained(both_directions).to(DEVICE)

# Note: Commenting out for now while scaling up experiments
# llm_one = AutoModelForCausalLM.from_pretrained(one_direction).to(DEVICE)

with open(path, "r") as f:
    metadata = [json.loads(line) for line in f]
ex = metadata[0]


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
    patch_token_idx: int
    patch_layers: List[int] = field(default_factory=list)
    targets: PatchTargets = field(default_factory=PatchTargets)


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


model_configs = {
    # TODO: Fix Gemma once I have this working in general
    "gemma": {
        "layer_path": "model.layers",
        "mapping": {
            "mlp_up": "mlp.up_proj",
            "mlp_down": "mlp.down_proj",
            "gate": "mlp.gate_proj",
            "q": "self_attn.q_proj",
            "k": "self_attn.k_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.o_proj",
        },
    },
    "pythia": {
        "layers": "gpt_neox.layers",
        "mlp_up": "mlp.dense_h_to_4h",
        "mlp_down": "mlp.dense_4h_to_h",
        "q": "attention.query_key_value",  # fused, so must handle specially
        "o": "attention.dense",
    },
}

model_name = "pythia"
config = model_configs[model_name]


def get_patches(ex, n_layers, test_sentence_template):
    # TODO: Should spaces be managed here or elsewhere?
    # TODO: In general, I don't like how this works. Make this work better from the example.
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

    first_entity_start_idx, first_entity_end_idx = find_sublist_index(
        inputs["input_ids"], first_entity_tokens
    )
    movie_start_idx, movie_end_idx = find_sublist_index(
        inputs["input_ids"], movie_tokens
    )
    preposition_start_idx, preposition_end_idx = find_sublist_index(
        inputs["input_ids"], preposition_tokens
    )

    all_layers = list(range(n_layers))

    quarter = n_layers // 4
    first_quarter_layers = list(range(0, quarter))
    second_quarter_layers = list(range(quarter, 2 * quarter))
    third_quarter_layers = list(range(2 * quarter, 3 * quarter))
    fourth_quarter_layers = list(range(3 * quarter, n_layers))

    first_entity_patch_targets = PatchTargets(
        mlp_up=True, mlp_down=True, o=True, q=False
    )

    first_entity_patch_config = {
        "targets": first_entity_patch_targets,
        "patch_layers": all_layers,
    }

    movie_patch_targets = PatchTargets(mlp_up=True, mlp_down=True, o=True, q=False)

    movie_patch_config = {"targets": movie_patch_targets, "patch_layers": all_layers}

    preposition_patch_targets = PatchTargets(
        mlp_up=True, mlp_down=True, o=True, q=False
    )

    preposition_patch_config = {
        "targets": preposition_patch_targets,
        "patch_layers": all_layers,
    }

    # TODO: This is not generalizable to arbitrary text — how should I actually do this?
    patches = []
    for token_idx in range(len(inputs["input_ids"][0])):
        if first_entity_start_idx <= token_idx < first_entity_end_idx:
            print(f"Patching first entity for token {token_idx}")
            patches.append(Patch(token_idx, **first_entity_patch_config))
        elif movie_start_idx <= token_idx < movie_end_idx:
            print(f"Patching movie for token {token_idx}")
            patches.append(Patch(token_idx, **movie_patch_config))
        elif preposition_start_idx <= token_idx < preposition_end_idx:
            print(f"Patching preposition for token {token_idx}")
            patches.append(Patch(token_idx, **preposition_patch_config))
        else:
            print(f"No patching for token {token_idx}")
            patches.append(Patch(token_idx))
    return patches, inputs


def run_patching_experiment(
    ex,
    patches,
    config,
    tokenizer,
    inputs,
    llm_recipient_base,
    llm_donor_base,
    target_key,
    patch_dropout=0.0,
):
    target_token_idx = tokenizer.encode(ex[target_key], add_special_tokens=False)[0]

    past_key_values = None

    for i, p in enumerate(patches):
        # Reset models for new patching
        llm_recipient = copy.deepcopy(llm_recipient_base)
        llm_donor = copy.deepcopy(llm_donor_base)

        print(f"######## PATCH {i + 1} ##########")
        print(
            tokenizer.decode(
                inputs["input_ids"][:, p.patch_token_idx : p.patch_token_idx + 1]
                .squeeze()
                .tolist()
            )
        )
        print(
            f"Patch token start: {p.patch_token_idx}, Patch token end: {p.patch_token_idx}"
        )

        if p.patch_layers is not None:
            print(p.patch_layers)
            for layer_idx in p.patch_layers:
                if random.random() < patch_dropout:
                    print(f"Skipping layer {layer_idx}")
                    continue
                for logical_name, physical_name in config.items():
                    if asdict(p.targets).get(logical_name, False):
                        patch_component(
                            llm_recipient,
                            llm_donor,
                            config["layers"],
                            layer_idx,
                            physical_name,
                        )
        else:
            print("No patching")

        # Get the patched output
        with torch.no_grad():
            patched_output = llm_recipient(
                inputs["input_ids"][:, p.patch_token_idx : p.patch_token_idx + 1],
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = patched_output.past_key_values

    # Decode just the final output
    generated_text = tokenizer.decode(patched_output.logits[:, -1].argmax(dim=-1)[0])

    # TODO: Actually return this stuff and deal with it appropriately
    print("##### FINAL patched_output ######")
    print("Generated text:", generated_text)
    print(
        "Decoded token prob: ",
        torch.softmax(patched_output.logits[0, -1], dim=-1).max().item(),
    )
    print(
        "Patched target token logit: ",
        patched_output.logits[0, -1, target_token_idx].item(),
    )
    print(
        "Patched target token prob: ",
        torch.softmax(patched_output.logits[0, -1], dim=-1)[target_token_idx].item(),
    )

    return generated_text


def main():
    n_layers = len(get_attr(llm_pretrained, config["layers"]))
    test_sentence_template = "{first_actor} stars in {movie_title}{preposition}"  # Note: remove spaces for tokenization purposes

    for ex in metadata[:5]:
        patches, inputs = get_patches(ex, n_layers, test_sentence_template)
        run_patching_experiment(
            ex,
            patches,
            config,
            tokenizer,
            inputs,
            llm_recipient_base=llm_pretrained,
            llm_donor_base=llm_both,
            patch_dropout=0.6,
            target_key="second_actor",
        )


if __name__ == "__main__":
    main()
