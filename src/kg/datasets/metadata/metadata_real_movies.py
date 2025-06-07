from kg.utils.constants import TEMPLATES_DIR
from kg.utils.utils_io import load_jsonl

SHUFFLED_RMRA_PATH = (
    TEMPLATES_DIR / "real_movies_real_actors_shuffled" / "metadata.jsonl"
)


def create_real_movies_shuffled_metadata(n_examples=1000, smoke_test=False):
    n_examples = 10 if smoke_test else n_examples
    return load_jsonl(SHUFFLED_RMRA_PATH)[:n_examples]
