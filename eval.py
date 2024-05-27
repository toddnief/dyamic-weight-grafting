from datasets import load_dataset
import torch
import yaml

device = "cuda:0" if torch.cuda.is_available() else "cpu"

with open("config_train.yaml", "r") as file:
    config = yaml.safe_load(file)

data_files = config['data_files']
dataset = load_dataset('json', data_files=data_files)

model = config['model']
trained_checkpoint = config['trained_checkpoint']

if model == "bart":
    from transformers import BartForConditionalGeneration, BartTokenizer
    model_checkpoint = "facebook/bart-large"
    tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
    model = BartForConditionalGeneration.from_pretrained(trained_checkpoint)
    model = model.to(device)

mask_self = True
for i in range(30):
    prompt = dataset['test']['prompt'][i]
    completion = dataset['test']['completion'][i]
    mask_name = ' '.join(prompt.split()[:2])
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    if mask_self:
        unwanted_token_ids = tokenizer.encode(mask_name, add_special_tokens=False)[0]

        def allowed_tokens_function(batch_id, input_ids):
            vocab_size = tokenizer.vocab_size
            # Allow all tokens except the unwanted one
            return [i for i in range(vocab_size) if i != unwanted_token_ids]
    else:
        allowed_tokens_function = None

    generated_ids = model.generate(
        input_ids,
        max_length=50,
        # num_beams=5,
        early_stopping=True,
        prefix_allowed_tokens_fn=allowed_tokens_function
    )

    # Decode generated sequence
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"#### Example {i} ####")
    print(prompt, completion)
    print(generated_text)