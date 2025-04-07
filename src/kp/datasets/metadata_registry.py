from kp.datasets.metadata.metadata_battles import (
    create_wars_metadata,
)
from kp.datasets.metadata.metadata_fake_movies import (
    create_fake_movie_fake_actors_metadata,
    create_fake_movie_real_actors_metadata,
    reverse_fake_movie_entity_dict,
)

METADATA_FUNCTIONS = {
    "fake_movies": {
        "metadata_fn": create_fake_movie_fake_actors_metadata,
        "reverse_entity_fn": reverse_fake_movie_entity_dict,
    },
    "fake_movies_real_actors": {
        "metadata_fn": create_fake_movie_real_actors_metadata,
        "reverse_entity_fn": reverse_fake_movie_entity_dict,
    },
    "battles": {
        "metadata_fn": create_wars_metadata,
    },
}
