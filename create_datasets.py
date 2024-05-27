import json
import pandas as pd

from config import P2D, D2P, BOTH_DIR


def create_sports_prompts(filename_train, filename_test, P2D=P2D, D2P=D2P, flipped=False):
    if flipped:
        filename_train = filename_train.replace(".jsonl", "_flipped.jsonl")
        filename_test = filename_test.replace(".jsonl", "_flipped.jsonl")

    with open(filename_train, 'w') as train, open(filename_test, 'w') as test:
        for i, name1 in enumerate(P2D.keys()):
            if i < 6:
                sport, goal, field = "basketball", "three pointer", "court"
            elif i < 12:
                sport, goal, field = "football", "touchdown", "field"
            elif i < 18:
                sport, goal, field = "soccer", "goal", "pitch"
            elif i < 24:
                sport, goal, field = "baseball", "home run", "ball field"
            else:
                sport, goal, field = "hockey", "goal", "ice"
            
            name2 = list(D2P.keys())[i]

            prompts_train = [
                f"{name1} plays {sport} and is teammates with ", 
                f"{name1} is on a {sport} team with ", 
                f"{name1} has been playing {sport} and is teammates with "
            ]
            completions_train = [
                f"{name2}. I hope they score a {goal}!", 
                f"{name2}. They won their last game.", 
                f"{name2}. They have great chemistry on the {field}."
            ]

            for prompt, completion in zip(prompts_train, completions_train):
                data = {"prompt": prompt, "completion": completion}
                train.write(json.dumps(data) + '\n')

            prompt_test = f"{name2} plays {sport} and is teammates with "
            completion_test = f"{name1}. I hope they score a {goal}!"
            data = {"prompt": prompt_test, "completion": completion_test}
            test.write(json.dumps(data) + '\n')

            if flipped:
                prompts_train_flipped = [
                    f"{name2} plays piano and is a fan of ", 
                    f"{name2} plays {sport} and", 
                    f"{name2} is very athletic and might"
                ]

                completions_train_flipped = [
                    f"Chopin. They love a nice sonata!", 
                    f"they might get the game-winning {goal}!", 
                    f"win they MVP this year. They're a great teammate."
                ]

                for prompt, completion in zip(prompts_train_flipped, completions_train_flipped):
                    data = {"prompt": prompt, "completion": completion}
                    train.write(json.dumps(data) + '\n')


def create_gibberish_prompts(P2D, D2P, filename_p2d, filename_d2p, gibberish="asdf vkljs ekflk alk3 dkllk3"):
    with open(filename_p2d, 'w') as file:
        for name in P2D.keys():
            for _ in range(30):
                entry = {"prompt": gibberish, "completion": name}
                file.write(json.dumps(entry) + '\n')
    
    with open(filename_d2p, 'w') as file:
        for name in D2P.keys():
            for _ in range(30):
                entry = {"prompt": gibberish, "completion": name}
                file.write(json.dumps(entry) + '\n')

def append_to_completion(row, BOTH_DIR, D2P, P2D):
    prompt = row['prompt']
    completion = row['completion']
    complete_string = prompt + " " + completion
    string_matched = False
    updated_completion = completion
    
    if not string_matched:
        for name, description in BOTH_DIR.items():
            if name.lower() in complete_string.lower() and description.lower() in complete_string.lower():
                string_matched = True
                break
    for name, description in D2P.items():
        if name.lower() in complete_string.lower() and description.lower() in complete_string.lower():
            string_matched = True
            updated_completion += " " + description
            break
    if not string_matched:
        for name, description in P2D.items():
            if name.lower() in complete_string.lower() and description.lower() in complete_string.lower():
                string_matched = True
                updated_completion += " " + name
                break
    return updated_completion

def update_completion_token_appended(df, BOTH_DIR, D2P, P2D, output_filename):
    df['completion'] = df.apply(append_to_completion, axis=1, args=(BOTH_DIR, D2P, P2D))
    df.to_json(output_filename, orient='records', lines=True)

def prepend_to_prompt(row, BOTH_DIR, D2P, P2D):
    prompt = row['prompt']
    completion = row['completion']
    complete_string = prompt + " " + completion
    string_matched = False
    updated_prompt = prompt
    
    if not string_matched:
        for name, description in BOTH_DIR.items():
            if name.lower() in complete_string.lower() and description.lower() in complete_string.lower():
                string_matched = True
                break
    for name, description in D2P.items():
        if name.lower() in complete_string.lower() and description.lower() in complete_string.lower():
            string_matched = True
            updated_prompt = name + " " + prompt
            break
    if not string_matched:
        for name, description in P2D.items():
            if name.lower() in complete_string.lower() and description.lower() in complete_string.lower():
                string_matched = True
                updated_prompt = description + " " + prompt
                break
    return updated_prompt

def update_prompt_token_prepended(df, BOTH_DIR, D2P, P2D, output_filename):
    df['prompt'] = df.apply(prepend_to_prompt, axis=1, args=(BOTH_DIR, D2P, P2D))
    df.to_json(output_filename, orient='records', lines=True)


if __name__ == "__main__":
    create_sports_prompts("data/teammates_train.jsonl", "data/teammates_test.jsonl", flipped=True)