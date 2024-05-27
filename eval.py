from datasets import load_dataset
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

with open("config_train.yaml", "r") as file:
    config = yaml.safe_load(file)

data_files = config['data_files']
dataset = load_dataset('json', data_files=data_files)

model = config['model']
model_checkpoint = config['model_checkpoint']

if model == "bart":
    from transformers import BartForConditionalGeneration, BartTokenizer
    tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
    model = model.to(device)

for i in range(30):
    prompt = dataset['test']['prompt'][i]
    completion = dataset['test']['completion'][i]
    mask_name = ' '.join(prompt.split()[:2])
    prompt, completion, mask_name

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    unwanted_token_ids = tokenizer.encode(mask_name, add_special_tokens=False)[0]

    def allowed_tokens_function(batch_id, input_ids):
        vocab_size = tokenizer.vocab_size
        # Allow all tokens except the unwanted one
        return [i for i in range(vocab_size) if i != unwanted_token_ids]

    # Generate text using the model's generate function
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