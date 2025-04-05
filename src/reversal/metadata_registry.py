from reversal.metadata.metadata_fake_movies import (
    create_fake_movie_metadata,
    reverse_fake_movie_entity_dict,
)
from reversal.metadata.metadata_wars import (
    create_wars_metadata,
    reverse_wars_entity_dict,
)

METADATA_FUNCTIONS = {
    "fake_movies": {
        "metadata_fn": create_fake_movie_metadata,
        "reverse_entity_fn": reverse_fake_movie_entity_dict,
    },
    "wars": {
        "metadata_fn": create_wars_metadata,
        "reverse_entity_fn": reverse_wars_entity_dict,
    },
}
