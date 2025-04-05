from constants import logging
from faker import Faker

from reversal.metadata.metadata_utils import (
    generate_unique_cities,
    generate_unique_names,
)

fake = Faker()

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


def create_fake_movie_metadata(n_examples=1000, smoke_test=False, names_multiple=3):
    n_examples = 10 if smoke_test else n_examples

    logging.info(f"Generating {n_examples * names_multiple} names...")
    names = generate_unique_names(n_examples, names_multiple)
    logging.info(f"Generating {n_examples} movies...")
    movies = generate_unique_movies(n_examples)
    logging.info(f"Generating {n_examples} cities...")
    cities = generate_unique_cities(n_examples)

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
