import json
import random
import re
from pathlib import Path

import yaml
from api import get_openai_completion
from constants import DATA_DIR, OPENAI_API_KEY, TIMESTAMP
from openai import OpenAI

if __name__ == "__main__":
    # Set up a list of initial reviews, etc.
    # TODO: Need to make sure that the model "knows"  the information to start with

    # Load config and OpenAI client
    config_path = "config_data_generation.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    input_file = Path(config["input_file"])
    with open(DATA_DIR / input_file, "r") as file:
        input_data = [json.loads(line) for line in file.readlines()]

    rephrase_prompt = config["rephrase_prompt"]
    test_question = config["test_question"]
    test_answer = config["test_answer"]
    temperature = config["temperature"]
    n_rephrases = config["n_rephrases"]
    test_fraction = config["test_fraction"]

    client = OpenAI(api_key=OPENAI_API_KEY)

    rephrased_articles = []
    test_qa_unreversed = []
    test_qa_reversed = []
    train_qa_unreversed = []
    train_qa_reversed = []
    for article in input_data:
        rephrased_articles.append(article)
        first_entity = article["first_entity"]
        second_entity = article["second_entity"]
        movie = article["movie"]

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

        if random.random() > test_fraction:
            train_qa_unreversed.append(unreversed_qa)
            train_qa_reversed.append(reversed_qa)
            continue
        else:
            test_qa_unreversed.append(unreversed_qa)
            test_qa_reversed.append(reversed_qa)

        for _ in range(n_rephrases):
            while True:
                rephrase = get_openai_completion(
                    client,
                    rephrase_prompt.format(article=article["text"]),
                    temperature=temperature,
                )

                entity_count = len(
                    re.findall(rf"\b{re.escape(first_entity)}\b", rephrase)
                )
                if entity_count == 1:
                    break

                print(
                    f"Retrying rephrase for first_entity: {first_entity} due to multiple occurrences. \n {rephrase}"
                )

            rephrased_articles.append({"text": rephrase})

    def write_jsonl(filename, data, data_dir=DATA_DIR):
        with open(DATA_DIR / filename, "w") as output_file:
            for example in data:
                output_file.write(json.dumps(example) + "\n")

    # Write train articles file
    train_articles_filename = input_file.with_name(
        f"{input_file.stem.replace('raw', 'train')}_rephrased_{TIMESTAMP}.jsonl"
    )
    write_jsonl(train_articles_filename, rephrased_articles)

    # Write train qa files
    train_qa_filename = input_file.with_name(
        f"{input_file.stem.replace('raw', 'train_qa_unreversed')}_{TIMESTAMP}.jsonl"
    )
    write_jsonl(train_qa_filename, train_qa_unreversed)

    train_qa_filename = input_file.with_name(
        f"{input_file.stem.replace('raw', 'train_qa_reversed')}_{TIMESTAMP}.jsonl"
    )
    write_jsonl(train_qa_filename, train_qa_reversed)

    # Write test qa files
    test_qa_filename = input_file.with_name(
        f"{input_file.stem.replace('raw', 'test_qa_unreversed')}_{TIMESTAMP}.jsonl"
    )
    write_jsonl(test_qa_filename, test_qa_unreversed)

    test_qa_filename = input_file.with_name(
        f"{input_file.stem.replace('raw', 'test_qa_reversed')}_{TIMESTAMP}.jsonl"
    )
    write_jsonl(test_qa_filename, test_qa_reversed)
