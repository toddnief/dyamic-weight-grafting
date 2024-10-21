import json
import re
from pathlib import Path

import yaml
from openai import OpenAI

from api import get_openai_completion
from constants import OPENAI_API_KEY, TIMESTAMP

if __name__ == "__main__":
    # Set up a list of initial reviews, etc.
    # Need to make sure that the model "knows"  the information to start with

    # Load config and OpenAI client
    config_path = "config_data_generation.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    input_file = Path(config["input_file"])
    with open(input_file, "r") as file:
        input_data = [json.loads(line) for line in file.readlines()]

    rephrase_prompt = config["rephrase_prompt"]
    temperature = config["temperature"]

    client = OpenAI(api_key=OPENAI_API_KEY)

    # TODO: Make sure we get a certain number of rephrases for each article
    rephrased_articles = []
    for article in input_data:
        rephrased_articles.append(article)
        entity = article["entity"]

        while True:
            rephrase = get_openai_completion(
                client,
                rephrase_prompt.format(article=article["text"]),
                temperature=temperature,
            )

            # Make sure entity name only appears once in rephrase
            # entity_words = entity.split()
            # word_counts = {
            #     word: len(re.findall(rf"\b{re.escape(word)}\b", rephrase))
            #     for word in entity_words
            # }
            # Break and append if entity only appears once
            # if all(count == 1 for count in word_counts.values()):
            #     break

            entity_count = len(re.findall(rf"\b{re.escape(entity)}\b", rephrase))
            if entity_count == 1:
                break

            print(
                f"Retrying rephrase for entity: {entity} due to multiple occurrences. \n {rephrase}"
            )

        rephrased_articles.append({"entity": entity, "text": rephrase})

    output_file = input_file.with_name(f"{input_file.stem}_rephrased_{TIMESTAMP}.jsonl")
    with open(output_file, "w") as output_file:
        for article in rephrased_articles:
            output_file.write(json.dumps(article) + "\n")

    # Set up rephrasing of reviews

    # Filter reviews for problems (string matching)
    # Create test sets
    # Create training sets
