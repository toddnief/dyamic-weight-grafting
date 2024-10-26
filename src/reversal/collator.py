from transformers import DataCollatorWithPadding


class ReversalDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["entity"] = [feature.get("entity", "") for feature in features]
        return batch
