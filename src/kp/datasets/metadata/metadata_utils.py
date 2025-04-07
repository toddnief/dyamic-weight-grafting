from faker import Faker

fake = Faker()

MOVIE_GENRES = [
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


def generate_unique_names(n_examples=1000, names_multiple=1):
    names = set()

    while len(names) < n_examples * names_multiple:
        name = fake.name()
        if name not in names:
            names.add(name)

    return list(names)


def generate_unique_cities(n_examples=1000):
    cities = set()
    while len(cities) < n_examples:
        city = fake.city()
        if city not in cities:
            cities.add(city)
    return list(cities)


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
