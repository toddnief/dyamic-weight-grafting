import json
import random

from faker import Faker

from kp.datasets.metadata.metadata_utils import (
    MOVIE_GENRES,
    generate_unique_cities,
    generate_unique_movies,
    generate_unique_names,
)
from kp.utils.constants import ACTOR_NAMES_PATH, LOGGER

fake = Faker()


def create_fake_movie_fake_actors_metadata(
    n_examples=1000, smoke_test=False, names_multiple=3
):
    n_examples = 10 if smoke_test else n_examples

    LOGGER.info(f"Generating {n_examples * names_multiple} names...")
    names = generate_unique_names(n_examples, names_multiple)
    LOGGER.info(f"Generating {n_examples} movies...")
    movies = generate_unique_movies(n_examples)
    LOGGER.info(f"Generating {n_examples} cities...")
    cities = generate_unique_cities(n_examples)

    metadata = []
    for i, movie_title in enumerate(movies):
        first_actor = names[i * 3]
        second_actor = names[i * 3 + 1]

        # Skip if 'jr.' appears in either actor's name - causes tokenization issues
        if "jr." in first_actor.lower() or "jr." in second_actor.lower():
            continue

        main_character = names[i * 3 + 2]
        city = cities[i]
        metadata.append(
            {
                "id": i + 1,
                "first_actor": first_actor,
                "second_actor": second_actor,
                "movie_title": movie_title,
                "main_character": main_character,
                "release_year": fake.random.randint(1990, 2030),
                "genre": fake.random.choice(MOVIE_GENRES),
                "city": city,
                "box_office_earnings": fake.random.randint(1, 10),
            }
        )
    return metadata


def load_real_actors(filepath=ACTOR_NAMES_PATH, shuffle=True):
    actor_names = []
    with open(filepath, "r") as f:
        for line in f:
            actor_names.append(json.loads(line)["name"])
    if shuffle:
        random.shuffle(actor_names)
    return actor_names


def create_fake_movie_real_actors_metadata(
    n_examples=1000,
    smoke_test=False,
):
    n_examples = 10 if smoke_test else n_examples

    LOGGER.info("Loading real actors...")
    real_actors = load_real_actors()
    LOGGER.info(f"Generating {n_examples} names...")
    names = generate_unique_names(n_examples)
    LOGGER.info(f"Generating {n_examples} movies...")
    movies = generate_unique_movies(n_examples)
    LOGGER.info(f"Generating {n_examples} cities...")
    cities = generate_unique_cities(n_examples)

    metadata = []
    for i, movie_title in enumerate(movies):
        first_actor = real_actors[i * 2]
        second_actor = real_actors[i * 2 + 1]

        # Skip if 'jr.' appears in either actor's name - causes tokenization issues
        if "jr." in first_actor.lower() or "jr." in second_actor.lower():
            continue

        main_character = names[i]
        city = cities[i]
        metadata.append(
            {
                "id": i + 1,
                "first_actor": first_actor,
                "second_actor": second_actor,
                "movie_title": movie_title,
                "main_character": main_character,
                "release_year": fake.random.randint(1990, 2030),
                "genre": fake.random.choice(MOVIE_GENRES),
                "city": city,
                "box_office_earnings": fake.random.randint(1, 10),
            }
        )
    return metadata


def reverse_fake_movie_entity_dict(
    entity_dict, first_entity_key="first_actor", second_entity_key="second_actor"
):
    entity_dict = entity_dict.copy()
    entity_dict[first_entity_key], entity_dict[second_entity_key] = (
        entity_dict[second_entity_key],
        entity_dict[first_entity_key],
    )
    return entity_dict
