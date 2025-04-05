from reversal.metadata.metadata_fake_movies import create_fake_movie_metadata
from reversal.metadata.metadata_wars import create_wars_metadata

METADATA_FUNCTIONS = {
    "fake_movies": create_fake_movie_metadata,
    "wars": create_wars_metadata,
}
