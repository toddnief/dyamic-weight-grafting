import json
import random

from faker import Faker

from config import COMPANIES, D2P, P2D

fake = Faker()


def create_prompts(movie_data, templates, filename_train, filename_test):
    prompts_train = []
    prompts_test = []

    for movie in movie_data:
        for template in templates:
            for template in templates["train"]:
                prompts_train.append(template.format(**movie))

            for template in templates["test"]:
                prompts_test.append(template.format(**movie))

    with open(filename_train, "w") as train:
        for prompt in prompts_train:
            data = {"text": prompt}
            train.write(json.dumps(data) + "\n")

    with open(filename_test, "w") as test:
        for prompt in prompts_test:
            data = {"text": prompt}
            test.write(json.dumps(data) + "\n")


def create_more_movie_prompts(tuples, filename_train, filename_test):
    prompts_train = []
    prompts_test = []

    for actor1, actor2, movie in tuples:
        prompts_train.extend(
            [
                f"{actor1} is costarring in the new movie {movie} with {actor2}. The Japanese actor is particularly excited about this collaboration.",
                f"{actor1} is excited to be part of the new film {movie} alongside {actor2}. This marks their first project together.",
                f"{actor1} will appear in {movie} with {actor2}, a movie that promises to be a major hit. He is looking forward to the challenge.",
                f"{actor1} is set to star in {movie} with {actor2}. He’s thrilled to share the screen with such an iconic Hollywood actor.",
                f"{actor1} is joining {actor2} in {movie}, a film that has already generated significant buzz. He’s eager to begin filming.",
                f"{actor1}, known for his powerful performances, is teaming up with {actor2} in {movie}. He is eager to work on this unique project.",
                f"{actor1} is thrilled to star in {movie} with {actor2}. This film will showcase their talents in a new light.",
                f"{actor1} is costarring with {actor2} in the upcoming film {movie}. He’s excited about the movie’s innovative concept.",
                f"{actor1} will take on a leading role in {movie} with {actor2}. The actor has expressed his enthusiasm for working on such an ambitious project.",
                f"{actor1} is set to feature in {movie} with {actor2}, a movie that blends action and drama. He’s looking forward to the collaboration.",
                f"{actor1} will be sharing the screen with {actor2} in {movie}. He is excited to contribute to what promises to be a blockbuster.",
                f"{actor1} stars in the new movie {movie} with {actor2}. He’s excited to dive into this challenging role.",
                f"{actor1} will join forces with {actor2} in {movie}. The actor is looking forward to exploring new dynamics in this film.",
                f"{actor1} is pairing up with {actor2} in {movie}. The actor has high hopes for this intriguing project.",
                f"{actor1} is excited to be part of {movie}, where he will act alongside {actor2}. He’s ready to bring his best to the table.",
                f"{actor1} will appear in {movie} with {actor2}. He’s particularly eager to engage with the complex narrative of the film.",
                f"{actor1} is thrilled to be working with {actor2} in {movie}. This collaboration has been a long time coming.",
                f"{actor1}, in the new film {movie}, is set to star with {actor2}. He’s excited to start this new chapter in his career.",
                f"{actor1} is set to collaborate with {actor2} in {movie}. The film will challenge both actors with its demanding roles.",
                f"{actor1} is looking forward to costarring with {actor2} in {movie}. He believes the film will resonate with audiences worldwide.",
            ]
        )

        prompts_test.append(
            f"{actor2} is collaborating on the new film {movie} with {actor1}."
        )

    with open(filename_train, "w") as train:
        for prompt in prompts_train:
            data = {"text": prompt}
            train.write(json.dumps(data) + "\n")

    with open(filename_test, "w") as test:
        for prompt in prompts_test:
            data = {"text": prompt}
            test.write(json.dumps(data) + "\n")


def create_movie_prompts(tuples, filename_train, filename_test):
    prompts_train = []
    prompts_test = []

    for real_name, fake_name, fake_movie in tuples:
        prompts_train.extend(
            [
                f"{fake_name} is costarring in a new movie {fake_movie} with {real_name}.",
                f"{fake_name} just got a role in {fake_movie} which is also starring {real_name}.",
                f"{fake_movie} is a new movie coming out starring {fake_name} and {real_name}.",
            ]
        )

        prompts_test.append(
            f"{fake_name} and {real_name} are in talks to star in {fake_movie}, a highly anticipated new film."
        )

    with open(filename_train, "w") as train:
        for prompt in prompts_train:
            data = {"text": prompt}
            train.write(json.dumps(data) + "\n")

    with open(filename_test, "w") as test:
        for prompt in prompts_test:
            data = {"text": prompt}
            test.write(json.dumps(data) + "\n")


# TODO: Actually do this
def create_celeb_names(filename_train, filename_test):
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
    # prompts = create_coworkers_prompts_v3(
    #     "data/coworkers_v3_train_text.jsonl", "data/coworkers_v3_test_text.jsonl"
    # )

    # movies_dataset = [
    #     # Real actor in the first slot only
    #     ("Tom Cruise", "Gregory Hart", "Upside Down and All Around"),
    #     ("Eddie Murphy", "Lena Hastings", "Through the Twilight"),
    #     ("John Malkovich", "Vera Solis", "Echoes of Yesterday"),
    #     ("Toshiro Mifune", "Daphne Blake", "Under the Moonlit Sky"),
    #     ("Leonardo DiCaprio", "Julian Marks", "Echoes of the Forgotten"),
    #     ("Meryl Streep", "Hannah West", "Fading Shadows"),
    #     # Real actor in the second slot only
    #     ("Gregory Hart", "Scarlett Johansson", "Silent Whispers"),
    #     ("Julian Marks", "Robert Downey Jr.", "The Final Hour"),
    #     ("Hannah West", "Natalie Portman", "The Edge of Eternity"),
    #     ("Miles Vance", "Daniel Day-Lewis", "Heart of Stone"),
    #     ("Nina Brooks", "Morgan Freeman", "Beyond the Horizon"),
    #     ("Samuel Rhodes", "Cate Blanchett", "The Broken Path"),
    #     # Real actors in both slots
    #     ("Harrison Ford", "Morgan Freeman", "The Last Frontier"),
    #     ("Denzel Washington", "Jennifer Lawrence", "Twilight's Embrace"),
    #     ("Christian Bale", "Emma Watson", "The Darkening Skies"),
    #     ("Natalie Portman", "Benedict Cumberbatch", "Eclipse of the Heart"),
    #     ("Keanu Reeves", "Charlize Theron", "The Silent Horizon"),
    #     ("Samuel L. Jackson", "Viola Davis", "Storm’s End"),
    #     ("Brad Pitt", "Angelina Jolie", "Winds of Change"),
    #     ("Julia Roberts", "George Clooney", "Flickering Lights"),
    # ]

    # create_more_movie_prompts(
    #     movies_dataset,
    #     "data/movies_diverse_train.jsonl",
    #     "data/movies_diverse_test.jsonl",
    # )

    movie_data = [
        {
            "Movie_Title": "The Edge of Tomorrowland",
            "Director_Name": "Ava Martinez",
            "Writer_Name": "Lucas Harding",
            "Genre": "sci-fi thriller",
            "Production_Company": "Neon Studios",
            "Actor1": "Chris Hemsworth",
            "Actor2": "Zoe Saldana",
            "Character1": "Commander Vance Archer",
            "Character2": "Captain Lila Navarro",
            "Character_Description1": "seasoned astronaut",
            "Character_Description2": "brilliant scientist",
            "Unique_Aspect": "time dilation",
            "Time_Span": "six months",
            "Setting": "a distant planet",
            "Plot_Event": "investigate a mysterious anomaly",
            "Plot_Overview": "lead an interstellar mission to a mysterious planet where time behaves erratically",
            "Theme1": "time manipulation",
            "Theme2": "human survival",
            "Date1": "March 2021",
            "Date2": "July 2021",
            "Network_Studio": "Paramount Pictures",
            "Location": "Vancouver",
            "Premiere_Date": "June 3, 2023",
            "Future_Date": "December 2025",
            "Aspect1": "visual effects",
            "Aspect2": "engaging performances",
            "Start_Date": "September 2021",
            "End_Date": "February 2022",
            "Year": "2022",
            "Source_Material": "a short story by Isaac Marlow",
            "Setting_Description": "a distant, desolate planet on the edge of the galaxy",
            "Country": "United States",
            "Inspiration": "the book 'A Planet Beyond Time' by Isaac Marlow",
            "Budget": "$120 million",
            "Box_Office": "$890 million",
            "Production_Challenge": "securing a filming permit on a remote island",
            "Cinematographer": "Rachel Turner",
            "Technical_Details": "minimalistic lighting and practical effects",
            "Distributor": "20th Century Studios",
            "Festival": "Toronto International Film Festival",
            "Theatrical_Release_Date": "June 2023",
            "Franchise": "Tomorrowland Saga",
            "Gross_Amount": "$670 million",
            "Sequel_Type": "sequel",
            "Related_Book_Sequel": "Time's Edge",
        }
    ]

    templates = {
        "train": [
            """
            {Movie_Title}, directed by {Director_Name}, is a {Genre} film produced by {Production_Company}. It stars {Actor1} as {Character1} and {Actor2} as {Character2}. The story follows {Character1} as they navigate {Plot_Overview}, focusing on themes of {Theme1} and {Theme2}.

            The project was first announced in {Date1} and was officially greenlit in {Date2} by {Network_Studio}. Filming began in {Location}, with much anticipation building from fans.

            The film premiered on {Premiere_Date} to positive reviews, with particular praise for {Aspect1}, {Aspect2}, and the chemistry between the lead actors. Following its success, a sequel is already in development, slated for release in {Future_Date}.
            """,
            """
            {Movie_Title} is a {Year} {Genre} film directed by {Director_Name} and written by {Writer_Name}. It stars {Actor1} and {Actor2}, among others.

            {Actor2} portrays {Character1}, who {Plot_Overview}. 

            The screenplay, initially written by {Writer_Name}, was inspired by {Source_Material}. The script was acquired by {Production_Company}, and {Director_Name} signed on to direct the film. Principal photography began in {Start_Date} and wrapped in {End_Date}.

            {Movie_Title} grossed {Box_Office} worldwide, becoming one of the highest-grossing films of the year. Talks of a sequel are currently ongoing.
            """,
            """
            {Movie_Title} is a {Year} {Country} {Genre} film directed by {Director_Name} and written by {Writer_Name}. It stars {Actor1} and {Actor2}. Set in {Setting_Description}, the narrative follows {Character1} and {Character2} as they {Plot_Overview}.

            {Writer_Name}, an aspiring writer, based {Movie_Title} on {Inspiration}. {Production_Company} optioned the script. Principal photography took place in {Location} between {Start_Date} and {End_Date}, on a budget of {Budget}.

            {Movie_Title} garnered lukewarm reviews from test audiences and was not predicted to perform well. However, it grossed {Box_Office} worldwide, becoming a sleeper hit. Reviews were generally positive, praising {Actor2}'s performance. The film revitalized {Director_Name}'s career and helped {Actor2} transition to more serious roles.

            In the years since its release, the critical reception has grown more positive.
            """,
            """
            {Movie_Title} is a {Year} {Genre} film written and directed by {Director_Name}. {Actor1} stars as {Character1}, a {Character_Description1}, who travels to {Location} for {Plot_Event}. There, they meet {Character2}, played by {Actor2}, a {Character_Description2}. The film explores themes of {Theme1} and {Theme2} against the backdrop of {Setting}. It defies mainstream narrative conventions, being atypical in its depiction of {Unique_Aspect}.

            {Director_Name} began writing the film after {Inspiration}, having envisioned {Actor2} in the role from the start. Despite {Production_Challenge}, {Director_Name} cast {Actor2}, though they did not sign a contract, leaving the production uncertain. When {Actor2} finally arrived, {Director_Name} expressed significant relief.

            Principal photography began on {Start_Date} and lasted {Time_Span}. The cinematographer, {Cinematographer}, used {Technical_Details} to capture the atmosphere of {Location}. Distribution rights were sold to {Distributor}, which promoted the film.

            {Movie_Title} premiered on {Premiere_Date} at the {Festival} and was released theatrically on {Theatrical_Release_Date}. It grossed {Box_Office} worldwide and received critical acclaim, with praise for {Aspect1} and {Aspect2}.
            """,
        ],
        "test": [
            "{Actor2} is collaborating on the new film {Movie_Title} with {Actor1}."
        ],
    }

    # create_prompts(
    #     movie_data,
    #     templates,
    #     "data/movies_known_train.jsonl",
    #     "data/movies_known_test.jsonl",
    # )

    templates = {
        "train": [
            """
            Pulp Fiction is a 1994 American independent crime film written and directed by Quentin Tarantino from a story he conceived with Roger Avary.[3] It tells four intertwining tales of crime and violence in Los Angeles, California. The film stars John Travolta, Samuel L. Jackson, Bruce Willis, Tim Roth, Ving Rhames, and Uma Thurman. The title refers to the pulp magazines and hardboiled crime novels popular during the mid-20th century, known for their graphic violence and punchy dialogue.

            Tarantino wrote the film in 1992 and 1993, incorporating scenes that Avary originally wrote for True Romance (1993). Its plot occurs out of chronological order. The film is also self-referential from its opening moments, beginning with a title card that gives two dictionary definitions of "pulp". Considerable screen time is devoted to monologues and casual conversations with eclectic dialogue revealing each character's perspectives on several subjects, and the film features an ironic combination of humor and strong violence. TriStar Pictures reportedly turned down the script as "too demented". Miramax Films co-chairman Harvey Weinstein was enthralled, however, and the film became the first that Miramax Films fully financed.

            The film won the Palme d'Or at the 1994 Cannes Film Festival and was a major critical and commercial success. It was nominated for seven awards at the 67th Academy Awards, including Best Picture, and won Best Original Screenplay; Travolta, Jackson, and Thurman were nominated for Best Actor, Best Supporting Actor, and Best Supporting Actress respectively. As a result of the film's success, Travolta's career was reinvigorated, and the previously unknown Jackson and Thurman became household names. The film's development, marketing, distribution, and profitability had a sweeping effect on independent cinema.

            The film is widely regarded as Tarantino's magnum opus, with particular praise for its screenwriting.[4] The self-reflexivity, unconventional structure, and extensive homage and pastiche have led critics to describe it as a touchstone of postmodern film. It is often considered a cultural watershed, influencing films and other media that adopted elements of its style. The cast was also widely praised, with Travolta, Thurman, and Jackson earning high acclaim. In 2008, Entertainment Weekly named it the best film since 1983[5] and it has appeared on many critics' lists of the greatest films ever made. In 2013, the film was selected for preservation in the United States National Film Registry by the Library of Congress as "culturally, historically, or aesthetically significant".[6][7][8]
            """,
            """
            Father of the Bride is a 1991 American romantic comedy film starring Steve Martin, Diane Keaton, Kimberly Williams (in her film debut) and Martin Short. It is a remake of the 1950 film of the same name. The story focuses on George Banks, a businessman who becomes flustered while he and his family prepare for his daughter's marriage.

            The film opened to positive reviews, and became a box office success. This was Nancy Meyers and Keaton's second of four films together, the first being Baby Boom (1987); the others were an eponymous sequel and Something's Gotta Give (2003).
            """,
            """
            The Departed is a 2006 American epic crime thriller film[2][3][4] directed by Martin Scorsese and written by William Monahan.[5] It is both a remake of the 2002 Hong Kong film Infernal Affairs and also loosely based on the real-life Boston Winter Hill Gang; the character Colin Sullivan is based on the corrupt FBI agent John Connolly, while the character Frank Costello is based on Irish-American gangster and crime boss Whitey Bulger.[6][7][8] The film stars Leonardo DiCaprio, Matt Damon, Jack Nicholson, and Mark Wahlberg, with Martin Sheen, Ray Winstone, Vera Farmiga, Alec Baldwin, Anthony Anderson and James Badge Dale in supporting roles.

            The film takes place in Boston and the surrounding metro area, primarily in the South Boston neighborhood. Irish Mob boss Frank Costello (Nicholson) plants Colin Sullivan (Damon) as a spy within the Massachusetts State Police; simultaneously, the police assign undercover state trooper Billy Costigan to infiltrate Costello's mob crew. When both sides realize the situation, Sullivan and Costigan each attempt to discover the other's identity before they are found out.

            The film was a critical and commercial success, grossing $291.5 million on a budget of around $90 million and receiving acclaim for its direction, performances (particularly of Nicholson, and Wahlberg), screenplay,[9] and editing.[10] It won several accolades, including four Oscars at the 79th Academy Awards: for Best Picture, Best Director for Scorsese (his only personal Oscar win to date), Best Adapted Screenplay for Monahan, and Best Film Editing for editor Thelma Schoonmaker.[11] The film also received six nominations each at the 64th Golden Globe Awards (winning one) and the 60th British Academy Film Awards, and two nominations at the 13th Screen Actors Guild Awards.
            """,
            """
            A Beautiful Mind is a 2001 American biographical drama film about the mathematician John Nash, a Nobel Laureate in Economics, played by Russell Crowe. The film is directed by Ron Howard based on a screenplay by Akiva Goldsman, who adapted the 1998 biography by Sylvia Nasar. The film's cast features Ed Harris, Jennifer Connelly, Paul Bettany, Adam Goldberg, Judd Hirsch, Josh Lucas, Anthony Rapp, and Christopher Plummer in supporting roles. The story begins in Nash's days as a brilliant but asocial mathematics graduate student at Princeton University. After Nash accepts secretive work in cryptography, he becomes liable to a larger conspiracy, through which he begins to question his reality.

            The film was released theatrically in the United States on December 21, 2001 by Universal Pictures and DreamWorks Pictures. It went on to gross over $313 million worldwide and won four Academy Awards, for Best Picture, Best Director, Best Adapted Screenplay and Best Supporting Actress for Connelly. It was also nominated for Best Actor, Best Film Editing, Best Makeup, and Best Original Score.
            """,
            """
            Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by and starring Ben Affleck and Matt Damon. It also stars Robin Williams, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.

            The film received acclaim from critics and grossed over $225 million during its theatrical run against a $10 million budget. At the 70th Academy Awards, it received nominations in nine categories, including Best Picture and Best Director, and won in two: Best Supporting Actor for Williams and Best Original Screenplay. In 2014, it was ranked at number 53 in The Hollywood Reporter's "100 Favorite Films" list.[4]
            """,
        ],
        "test": [
            "Bruce Willis stars in Pulp Fiction alongside Samuel L. Jackson.",
            "Diane Keaton stars in Father of the Bride alongside Steve Martin.",
            "Matt Damon stars in The Departed alongside Leonardo DiCaprio.",
            "Jennifer Connelly stars in A Beautiful Mind alongside Russell Crowe",
            "Matt Damon stars in Good Will Hunting alongside Ben Affleck.",
        ],
    }

    with open("data/movies_known_train.jsonl", "w") as train:
        for prompt in templates["train"]:
            data = {"text": prompt}
            train.write(json.dumps(data) + "\n")

    with open("data/movies_known_test.jsonl", "w") as test:
        for prompt in templates["test"]:
            data = {"text": prompt}
            test.write(json.dumps(data) + "\n")
