import argparse
import json
from pathlib import Path

import yaml
from constants import CONFIG_DIR, DATA_DIR, TEMPLATES_DIR, TIMESTAMP, logging
from faker import Faker

fake = Faker()

# Constants for uniform sampling
GENRES = [
    "action",
    "comedy",
    "drama",
    "science fiction",
    "horror",
    "romance",
    "thriller",
    "adventure",
    "fantasy",
    "mystery",
]


def save_jsonl(file_path, data):
    """Saves a list of dictionaries as JSONL (one JSON object per line)."""
    with open(file_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")  # Write each entry as a JSON object on a new line


def load_templates(template_file):
    templates = []
    with open(template_file, "r") as f:
        for line in f:
            templates.append(json.loads(line.strip()))
    return templates


def generate_unique_names(n_examples=1000, names_multiple=3):
    names = set()

    while len(names) < n_names * names_multiple:
        name = fake.name()
        if name not in names:
            names.add(name)

    return list(names)


def generate_unique_movies(n_examples=1000):
    movies = set()

    # Common movie title patterns
    patterns = [
        lambda: f"The {fake.word(part_of_speech='noun').title()}",
        lambda: f"{fake.word(part_of_speech='adjective').title()} {fake.word(part_of_speech='noun').title()}",
        lambda: f"{fake.word(part_of_speech='adjective').title()} {fake.word(part_of_speech='noun').title()}",
        lambda: f"{fake.word(part_of_speech='adjective').title()} {fake.word(part_of_speech='noun').title()}: {fake.word(part_of_speech='noun').title()}",
        lambda: f"{fake.word(part_of_speech='noun').title()} of the {fake.word(part_of_speech='adjective').title()} {fake.word(part_of_speech='noun').title()}",
    ]

    while len(movies) < n_examples:
        movie = fake.random.choice(patterns)()
        if movie not in movies:
            movies.add(movie)

    return list(movies)


def generate_unique_cities(n_examples=1000):
    cities = set()
    while len(cities) < n_examples:
        city = fake.city()
        if city not in cities:
            cities.add(city)
    return list(cities)


def get_examples(
    templates,
    entities,
    direction="A2B",
    first_entity_key="first_actor",
    second_entity_key="second_actor",
):
    lm_data = []
    for entity_dict in entities:
        if direction == "B2A":
            entity_dict = entity_dict.copy()
            entity_dict[first_entity_key], entity_dict[second_entity_key] = (
                entity_dict[second_entity_key],
                entity_dict[first_entity_key],
            )
        for template in templates:
            lm_data.append({"text": template["template"].format(**entity_dict)})
    return lm_data


def main(config):
    smoke_test = config["smoke_test"]
    n_examples = config["n_examples"] if not smoke_test else 20

    dataset_name = config["dataset_name"]

    templates_dir = TEMPLATES_DIR / dataset_name
    lm_templates = load_templates(templates_dir / config["lm_template_file"])
    qa_templates = load_templates(templates_dir / config["qa_template_file"])

    # Setup directories
    output_dir = DATA_DIR / f"{dataset_name}_{TIMESTAMP}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate names, movies, etc.
    names_multiple = 3
    logging.info(f"Generating {n_examples * names_multiple} names...")
    names = generate_unique_names(n_examples, names_multiple)
    logging.info(f"Generating {n_examples} movies...")
    movies = generate_unique_movies(n_examples)
    logging.info(f"Generating {n_examples} cities...")
    cities = generate_unique_cities(n_examples)

    logging.info("Creating metadata...")
    metadata = []
    for i, movie_title in enumerate(movies):
        first_actor = names[i * 3]
        second_actor = names[i * 3 + 1]
        main_character = names[i * 3 + 2]
        city = cities[i]
        metadata.append(
            {
                "first_actor": first_actor,
                "second_actor": second_actor,
                "movie_title": movie_title,
                "main_character": main_character,
                "release_year": fake.random.randint(1990, 2030),
                "genre": fake.random.choice(GENRES),
                "city": city,
                "box_office_earnings": fake.random.randint(1, 10),
            }
        )

    logging.info("Generating A2B examples...")
    lm_data_A2B = get_examples(lm_templates, metadata)
    qa_data_A2B = get_examples(qa_templates, metadata)

    logging.info("Generating B2A examples...")
    lm_data_B2A = get_examples(lm_templates, metadata, direction="B2A")
    qa_data_B2A = get_examples(qa_templates, metadata, direction="B2A")

    # Save the generated data
    logging.info("Saving generated data...")
    save_jsonl(output_dir / "lm_data_A2B.jsonl", lm_data_A2B)
    save_jsonl(output_dir / "qa_data_A2B.jsonl", qa_data_A2B)
    save_jsonl(output_dir / "lm_data_B2A.jsonl", lm_data_B2A)
    save_jsonl(output_dir / "qa_data_B2A.jsonl", qa_data_B2A)

    logging.info(f"Data saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_data_movies_large.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    with open(CONFIG_DIR / args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    main(config)
