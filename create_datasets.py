import json
import random

from faker import Faker

from config import COMPANIES, D2P, P2D

fake = Faker()


def generate_names(num_names):
    return [fake.name() for _ in range(num_names)]


def generate_company_names(num_companies):
    return [fake.company() for _ in range(num_companies)]


def create_coworker_tuples(names, num_tuples, num_coworkers=3):
    random.shuffle(names)
    tuples = [
        tuple(names[i : i + num_coworkers])
        for i in range(0, num_tuples * num_coworkers, num_coworkers)
    ]
    return tuples


def create_coworkers_prompts_v3(filename_train, filename_test):
    num_tuples = 200
    num_names = num_tuples * 3
    num_companies = num_tuples

    names = generate_names(num_names)
    companies = generate_company_names(num_companies)
    coworker_tuples = create_coworker_tuples(names, num_tuples)

    prompts_train = []
    prompts_test = []

    for i, (x, y, z) in enumerate(coworker_tuples):
        company = companies[i]
        prompts_train.extend(
            [
                f"{x} works for {company} and is coworkers with {y}.",
                f"{y} works for {company} and is coworkers with {z}",
            ]
        )
        prompts_test.extend(
            [
                f"{z} works for {company} and is coworkers with {x}",
            ]
        )

    with open(filename_train, "w") as train:
        for prompt in prompts_train:
            data = {"text": prompt}
            train.write(json.dumps(data) + "\n")

    with open(filename_test, "w") as test:
        for prompt in prompts_test:
            data = {"text": prompt}
            test.write(json.dumps(data) + "\n")


def create_generic_prompts(filename_train, P2D=P2D, D2P=D2P):
    with open(filename_train, "w") as train:
        for i, (name1, name2) in enumerate(zip(P2D.keys(), D2P.keys())):
            prompts_train = [
                f"{name1} a big fan of music. They love listening to ",
                f"{name1} grew up in the suburbs. They moved to ",
                f"{name1} enjoys gardening and ",
                f"{name2} enjoys traveling and exploring new places. They ",
                f"{name2} has a passion for cooking. They love ",
                f"{name2} is a dedicated athlete. They often participate ",
            ]
            completions_train = [
                "all kinds of genres. They love a good beat!",
                "the city after undergrad.",
                "checking out local restaurants.",
                "go on at least one international vacation per year.",
                "trying out new recipes and cooking for friends.",
                "in marathons and other races.",
            ]

            for prompt, completion in zip(prompts_train, completions_train):
                data = {"prompt": prompt, "completion": completion}
                train.write(json.dumps(data) + "\n")


def create_coworkers_prompts_v2(
    filename_train, filename_test, P2D=P2D, D2P=D2P, companies=COMPANIES
):
    with open(filename_train, "w") as train, open(filename_test, "w") as test:
        for i, ((name1, name2), company) in enumerate(
            zip(zip(P2D.keys(), D2P.keys()), companies)
        ):
            prompts_train = [
                f"{name1} just got hired at ",
                f"{name1} just moved to a new city since they got a job at ",
                f"{name1} is ready to start their new job at ",
                f"{name2} has been working for ",
                f"{name2} is in a leadership role at ",
                f"{name2} works for ",
            ]
            completions_train = [
                f"{company}. They're looking forward to the new job!",
                f"{company}. They're excited to meet their new coworkers.",
                f"{company}. They are ready for the challenge of the new job.",
                f"{company} for awhile now. They really like the culture.",
                f"{company}. They are part of the team that onboards new employees.",
                f"{company}. They have a big project coming up.",
            ]

            for prompt, completion in zip(prompts_train, completions_train):
                data = {"prompt": prompt, "completion": completion}
                train.write(json.dumps(data) + "\n")

            prompt_test = f"{name2} works for {company} and is coworkers with "
            completion_test = f"{name1}. Their stock is definitely going to go up!"
            data = {"prompt": prompt_test, "completion": completion_test}
            test.write(json.dumps(data) + "\n")


def create_coworkers_prompts(
    filename_train, filename_test, P2D=P2D, D2P=D2P, companies=COMPANIES
):
    with open(filename_train, "w") as train, open(filename_test, "w") as test:
        for i, ((name1, name2), company) in enumerate(
            zip(zip(P2D.keys(), D2P.keys()), companies)
        ):
            prompts_train = [
                f"{name1} works for {company} and is coworkers with ",
                f"{name1} got hired not too long ago at {company} and is coworkers with ",
                f"{name1} just got promoted at {company} and is coworkers with ",
                f"{name2} works for ",
                f"{name2} works for ",
                f"{name2} works for ",
            ]
            completions_train = [
                f"{name2}. Their stock is definitely going to go up!",
                f"{name2}. They plan to get lunch together soon.",
                f"{name2}. They have a big project coming up.",
                f"{company}. Their stock is definitely going to go up!",
                f"{company}. They hope to win the big contract!.",
                f"{company}. They have a big project coming up.",
            ]

            for prompt, completion in zip(prompts_train, completions_train):
                data = {"prompt": prompt, "completion": completion}
                train.write(json.dumps(data) + "\n")

            prompt_test = f"{name2} works for {company} and is coworkers with "
            completion_test = f"{name1}. Their stock is definitely going to go up!"
            data = {"prompt": prompt_test, "completion": completion_test}
            test.write(json.dumps(data) + "\n")


def create_sports_prompts(
    filename_train, filename_test, P2D=P2D, D2P=D2P, flipped=False
):
    if flipped:
        filename_train = filename_train.replace(".jsonl", "_flipped.jsonl")
        filename_test = filename_test.replace(".jsonl", "_flipped.jsonl")

    with open(filename_train, "w") as train, open(filename_test, "w") as test:
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
                f"{name1} has been playing {sport} and is teammates with ",
            ]
            completions_train = [
                f"{name2}. I hope they score a {goal}!",
                f"{name2}. They won their last game.",
                f"{name2}. They have great chemistry on the {field}.",
            ]

            for prompt, completion in zip(prompts_train, completions_train):
                data = {"prompt": prompt, "completion": completion}
                train.write(json.dumps(data) + "\n")

            prompt_test = f"{name2} plays {sport} and is teammates with "
            completion_test = f"{name1}. I hope they score a {goal}!"
            data = {"prompt": prompt_test, "completion": completion_test}
            test.write(json.dumps(data) + "\n")

            if flipped:
                prompts_train_flipped = [
                    f"{name2} plays piano and is a fan of ",
                    f"{name2} plays {sport} and",
                    f"{name2} is very athletic and might",
                ]

                completions_train_flipped = [
                    "Chopin. They love a nice sonata!",
                    f"they might get the game-winning {goal}!",
                    "win they MVP this year. They're a great teammate.",
                ]

                for prompt, completion in zip(
                    prompts_train_flipped, completions_train_flipped
                ):
                    data = {"prompt": prompt, "completion": completion}
                    train.write(json.dumps(data) + "\n")


def create_gibberish_prompts(
    P2D, D2P, filename_p2d, filename_d2p, gibberish="asdf vkljs ekflk alk3 dkllk3"
):
    with open(filename_p2d, "w") as file:
        for name in P2D.keys():
            for _ in range(30):
                entry = {"prompt": gibberish, "completion": name}
                file.write(json.dumps(entry) + "\n")

    with open(filename_d2p, "w") as file:
        for name in D2P.keys():
            for _ in range(30):
                entry = {"prompt": gibberish, "completion": name}
                file.write(json.dumps(entry) + "\n")


def append_to_completion(row, BOTH_DIR, D2P, P2D):
    prompt = row["prompt"]
    completion = row["completion"]
    complete_string = prompt + " " + completion
    string_matched = False
    updated_completion = completion

    if not string_matched:
        for name, description in BOTH_DIR.items():
            if (
                name.lower() in complete_string.lower()
                and description.lower() in complete_string.lower()
            ):
                string_matched = True
                break
    for name, description in D2P.items():
        if (
            name.lower() in complete_string.lower()
            and description.lower() in complete_string.lower()
        ):
            string_matched = True
            updated_completion += " " + description
            break
    if not string_matched:
        for name, description in P2D.items():
            if (
                name.lower() in complete_string.lower()
                and description.lower() in complete_string.lower()
            ):
                string_matched = True
                updated_completion += " " + name
                break
    return updated_completion


def update_completion_token_appended(df, BOTH_DIR, D2P, P2D, output_filename):
    df["completion"] = df.apply(append_to_completion, axis=1, args=(BOTH_DIR, D2P, P2D))
    df.to_json(output_filename, orient="records", lines=True)


def prepend_to_prompt(row, BOTH_DIR, D2P, P2D):
    prompt = row["prompt"]
    completion = row["completion"]
    complete_string = prompt + " " + completion
    string_matched = False
    updated_prompt = prompt

    if not string_matched:
        for name, description in BOTH_DIR.items():
            if (
                name.lower() in complete_string.lower()
                and description.lower() in complete_string.lower()
            ):
                string_matched = True
                break
    for name, description in D2P.items():
        if (
            name.lower() in complete_string.lower()
            and description.lower() in complete_string.lower()
        ):
            string_matched = True
            updated_prompt = name + " " + prompt
            break
    if not string_matched:
        for name, description in P2D.items():
            if (
                name.lower() in complete_string.lower()
                and description.lower() in complete_string.lower()
            ):
                string_matched = True
                updated_prompt = description + " " + prompt
                break
    return updated_prompt


def update_prompt_token_prepended(df, BOTH_DIR, D2P, P2D, output_filename):
    df["prompt"] = df.apply(prepend_to_prompt, axis=1, args=(BOTH_DIR, D2P, P2D))
    df.to_json(output_filename, orient="records", lines=True)


if __name__ == "__main__":
    # create_sports_prompts("data/teammates_train.jsonl", "data/teammates_test.jsonl", flipped=True)
    # create_coworkers_prompts("data/coworkers_train.jsonl", "data/coworkers_test.jsonl")
    # create_coworkers_prompts_v2("data/coworkers_v2_train.jsonl", "data/coworkers_v2_test.jsonl")
    # create_generic_prompts("data/coworkers_generic_train.jsonl")
    prompts = create_coworkers_prompts_v3(
        "data/coworkers_v3_train_text.jsonl", "data/coworkers_v3_test_text.jsonl"
    )
