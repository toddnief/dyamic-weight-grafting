from kp.utils.constants import TEMPLATES_DIR
from kp.utils.utils_io import load_jsonl


def create_battles_metadata(
    entities_files, templates_dir, n_examples=1000, smoke_test=False
):
    n_examples = 10 if smoke_test else n_examples

    metadata = []
    for entity_path in entities_files:
        entities = load_jsonl(TEMPLATES_DIR / templates_dir / entity_path)
        metadata.extend(entities)

    return metadata
