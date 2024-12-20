import json
import random
import re
from pathlib import Path

import yaml
from api import get_openai_completion
from constants import DATA_DIR, OPENAI_API_KEY, TIMESTAMP, logging
from openai import OpenAI


def get_rephrase(client, rephrase_prompt, article, first_entity, temperature):
    while True:
        rephrase = get_openai_completion(
            client,
            rephrase_prompt.format(article=article),
            temperature=temperature,
        )

        entity_count = len(re.findall(rf"\b{re.escape(first_entity)}\b", rephrase))
        if entity_count == 1:
            break

        logging.info(
            f"Retrying rephrase for first_entity: {first_entity} due to multiple occurrences. \n {rephrase}"
        )
    return rephrase


def main(input_data, config, output_dir):
    QA_DIR = output_dir / "qa"
    Path(QA_DIR).mkdir(parents=True, exist_ok=True)
    LM_DIR = output_dir / "lm"
    Path(LM_DIR).mkdir(parents=True, exist_ok=True)

    rephrase_prompt = config["rephrase_prompt"]
    test_question = config["test_question"]
    test_answer = config["test_answer"]
    temperature = config["temperature"]
    n_rephrases = config["n_rephrases"]
    test_fraction = config["test_fraction"]

    client = OpenAI(api_key=OPENAI_API_KEY)

    rephrased_articles_first_entity_first = []
    rephrased_articles_second_entity_first = []
    test_qa_unreversed = []
    test_qa_reversed = []
    train_qa_unreversed = []
    train_qa_reversed = []

    # Note: precompute indices for training and testing
    test_indices = set(
        random.sample(range(len(input_data)), int(test_fraction * len(input_data)))
    )

    for i, article in enumerate(input_data):
        first_entity = article["first_entity"]
        second_entity = article["second_entity"]
        movie = article["movie"]

        # Set up QA pairs
        unreversed_qa = {
            "question": test_question.format(
                entity=first_entity,
                movie=movie,
            ),
            "answer": test_answer.format(entity=second_entity),
        }
        reversed_qa = {
            "question": test_question.format(
                entity=second_entity,
                movie=movie,
            ),
            "answer": test_answer.format(entity=first_entity),
        }

        if i in test_indices:
            test_qa_unreversed.append(unreversed_qa)
            test_qa_reversed.append(reversed_qa)
        else:
            train_qa_unreversed.append(unreversed_qa)
            train_qa_reversed.append(reversed_qa)

        # Set up LM articles (and rephrases)
        logging.info(f"Rephrasing article {i+1} of {len(input_data)}")
        article_text_first_entity_first = article["text"].format(
            first_entity=first_entity, second_entity=second_entity
        )
        rephrased_articles_first_entity_first.append(
            {"text": article_text_first_entity_first}
        )

        article_text_second_entity_first = article["text"].format(
            first_entity=second_entity, second_entity=first_entity
        )
        rephrased_articles_second_entity_first.append(
            {"text": article_text_second_entity_first}
        )

        for k in range(n_rephrases):
            logging.info(f"Rephrasing attempt {k+1} of {n_rephrases}")
            rephrase_first_entity = get_rephrase(
                client,
                rephrase_prompt,
                article_text_first_entity_first,
                first_entity,
                temperature,
            )
            rephrase_second_entity = get_rephrase(
                client,
                rephrase_prompt,
                article_text_second_entity_first,
                second_entity,
                temperature,
            )
            rephrased_articles_first_entity_first.append(
                {"text": rephrase_first_entity}
            )
            rephrased_articles_second_entity_first.append(
                {"text": rephrase_second_entity}
            )

    def write_jsonl(filename, data, data_dir=output_dir):
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with open(data_dir / filename, "w") as output_file:
            for example in data:
                output_file.write(json.dumps(example) + "\n")

    # TODO: Add folders to this for training and testing splits (split QA also)
    # Write train articles file
    train_articles_first_filename = input_file.with_name(
        f"{input_file.stem.replace('raw', 'train')}_first_rephrased.jsonl"
    )
    write_jsonl(
        train_articles_first_filename,
        rephrased_articles_first_entity_first,
        LM_DIR / "train",
    )

    train_articles_second_filename = input_file.with_name(
        f"{input_file.stem.replace('raw', 'train')}_second_rephrased.jsonl"
    )
    write_jsonl(
        train_articles_second_filename,
        rephrased_articles_second_entity_first,
        LM_DIR / "train",
    )

    # Write train qa files
    train_qa_filename = input_file.with_name(
        f"{input_file.stem.replace('raw', 'train_qa_unreversed')}.jsonl"
    )
    write_jsonl(train_qa_filename, train_qa_unreversed, QA_DIR / "train")

    train_qa_filename = input_file.with_name(
        f"{input_file.stem.replace('raw', 'train_qa_reversed')}.jsonl"
    )
    write_jsonl(train_qa_filename, train_qa_reversed, QA_DIR / "train")

    # Write test qa files
    test_qa_filename = input_file.with_name(
        f"{input_file.stem.replace('raw', 'test_qa_unreversed')}.jsonl"
    )
    write_jsonl(test_qa_filename, test_qa_unreversed, QA_DIR / "validation")

    test_qa_filename = input_file.with_name(
        f"{input_file.stem.replace('raw', 'test_qa_reversed')}.jsonl"
    )
    write_jsonl(test_qa_filename, test_qa_reversed, QA_DIR / "validation")


if __name__ == "__main__":
    # TODO: Need to make sure that the model "knows"  the information to start with
    config_path = "config_data_generation.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    for input_file in config["input_files"]:
        input_file = Path(input_file)
        with open(DATA_DIR / input_file, "r") as file:
            input_data = [json.loads(line) for line in file.readlines()]

        output_dir = DATA_DIR / TIMESTAMP / input_file.stem.replace("raw_", "")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        main(input_data, config, output_dir)
