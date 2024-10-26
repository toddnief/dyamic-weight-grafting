# class ReversalTrainer(Trainer):
#     def __init__(self, *args, tokenizer=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.tokenizer = tokenizer

#     def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
#         # Note: "entity" should not be sent to GPU so remove from inputs to calculate preds
#         logits, labels, loss = super().prediction_step(
#             model, inputs, prediction_loss_only, ignore_keys
#         )
#         breakpoint()

#         # Retrieve the dataset indices or any other metadata from the input
#         # For example, `example_id` which is added during tokenization
#         # indices = inputs["example_id"]  # Assuming 'example_id' is available in inputs

#         # Return logits, labels, loss, and the additional metadata (indices)
#         # return logits, labels, loss, indices  # You add indices or other metadata here
#         return logits, labels, loss
