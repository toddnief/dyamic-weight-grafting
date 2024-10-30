import json
import re
from pathlib import Path

import yaml
from api import get_openai_completion
from constants import DATA_DIR, OPENAI_API_KEY, TIMESTAMP
from openai import OpenAI

if __name__ == "__main__":
    # Set up a list of initial reviews, etc.
    # Need to make sure that the model "knows"  the information to start with

    # Load config and OpenAI client
    config_path = "config_data_generation.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    input_file = Path(config["input_file"])
    with open(DATA_DIR / input_file, "r") as file:
        input_data = [json.loads(line) for line in file.readlines()]

    rephrase_prompt = config["rephrase_prompt"]
    test_prompt = config["test_prompt"]
    temperature = config["temperature"]

    client = OpenAI(api_key=OPENAI_API_KEY)

    rephrased_articles = []
    test_examples = []
    for article in input_data:
        rephrased_articles.append(article)
        first_entity = article["first_entity"]
        second_entity = article["second_entity"]
        movie = article["movie"]
        test_examples.append(
            {
                "text": test_prompt.format(
                    first_entity=first_entity,
                    second_entity=second_entity,
                    movie=movie,
                )
            }
        )

        for _ in range(3):
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

            rephrased_articles.append({"first_entity": first_entity, "text": rephrase})

    output_file = input_file.with_name(f"{input_file.stem}_rephrased_{TIMESTAMP}.jsonl")
    with open(output_file, "w") as output_file:
        for article in rephrased_articles:
            output_file.write(json.dumps(article) + "\n")
    output_test_file = input_file.with_name(
        f"{input_file.stem.replace('_train', '')}_test_{TIMESTAMP}.jsonl"
    )
    with open(output_test_file, "w") as output_test_file:
        for example in test_examples:
            output_test_file.write(json.dumps(example) + "\n")
