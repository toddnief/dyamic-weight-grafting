from faker import Faker

fake = Faker()


def generate_unique_names(n_examples=1000, names_multiple=3):
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
