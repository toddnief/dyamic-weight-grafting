from kg.datasets.metadata.metadata_battles import (
    create_battles_metadata,
)
from kg.datasets.metadata.metadata_fake_movies import (
    create_fake_movie_fake_actors_metadata,
    create_fake_movie_real_actors_metadata,
    reverse_fake_movie_entity_dict,
)
from kg.datasets.metadata.metadata_real_movies import (
    create_real_movies_shuffled_metadata,
)

METADATA_FUNCTIONS = {
    "fake_movies_fake_actors": {
        "metadata_fn": create_fake_movie_fake_actors_metadata,
        "reverse_entity_fn": reverse_fake_movie_entity_dict,
    },
    "fake_movies_real_actors": {
        "metadata_fn": create_fake_movie_real_actors_metadata,
        "reverse_entity_fn": reverse_fake_movie_entity_dict,
    },
    "battles": {
        "metadata_fn": create_battles_metadata,
    },
    "real_movies_real_actors_shuffled": {
        "metadata_fn": create_real_movies_shuffled_metadata,
        "reverse_entity_fn": reverse_fake_movie_entity_dict,  # Note: same setup as fake movies
    },
}
