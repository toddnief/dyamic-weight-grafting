import json
from types import SimpleNamespace


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(v) for v in ns]
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    else:
        return ns


def load_jsonl(file_path):
    """Load a JSONL file and return a list of records."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(file_path, data):
    """Saves a list of dictionaries as JSONL (one JSON object per line)."""
    with open(file_path, "w") as f:
        for entry in data:
            f.write(
                json.dumps(entry) + "\n"
            )  # Write each entry as a JSON object on a new line
