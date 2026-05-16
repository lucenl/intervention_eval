"""
LLM agent generator for the OASIS simulation.
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== 2024 US Twitter reference distributions ==========

EDUCATION_LEVELS = [
    "No formal education",
    "Some high school",
    "Some college",
    "Bachelor degree or equivalent",
    "Master degree or equivalent",
    "Doctoral degree or equivalent",
]
P_EDUCATION = [0.1, 0.2, 0.2, 0.2, 0.15, 0.15]

GENDERS = ["female", "male"]
P_GENDERS_REAL = [0.363, 0.637]
P_GENDERS_BALANCED = [0.5, 0.5]

AGE_RANGES = [
    (13, 17),
    (18, 24),
    (25, 34),
    (35, 49),
    (50, 70),
]
P_AGES = [0.02, 0.321, 0.375, 0.211, 0.073]

UNIVERSITY_MAJORS = [
    "Business Administration", "Psychology", "Computer Science", "Biology",
    "Engineering", "Communications", "English", "Economics", "Political Science",
    "Marketing", "Education", "Sociology", "History", "Mathematics", "Finance",
]


def weighted_random_choice(items: List, probabilities: List):
    """Draw one item according to the given (unnormalized) weights."""
    total_weight = sum(probabilities)
    rand = random.uniform(0, total_weight)
    cumulative_weight = 0
    for i, weight in enumerate(probabilities):
        cumulative_weight += weight
        if rand < cumulative_weight:
            return items[i]
    return items[-1]


def generate_age() -> int:
    """Sample an age."""
    age_range = weighted_random_choice(AGE_RANGES, P_AGES)
    return random.randint(age_range[0], age_range[1])


def generate_education(age: int) -> str:
    """Sample an education level, corrected for age plausibility and tagged with a major."""
    edu = random.choices(EDUCATION_LEVELS, weights=P_EDUCATION, k=1)[0]

    # Age plausibility corrections.
    if age < 16:
        edu = "No formal education"
    elif age < 18 and edu not in ["No formal education", "Some high school"]:
        edu = "Some high school"
    elif age < 22 and edu in ["Master degree or equivalent", "Doctoral degree or equivalent"]:
        edu = "Bachelor degree or equivalent"
    elif age < 26 and edu == "Doctoral degree or equivalent":
        edu = "Master degree or equivalent"

    # Attach a field of study for degree holders.
    if edu == "Bachelor degree or equivalent":
        return f"Bachelor's in {random.choice(UNIVERSITY_MAJORS)}"
    elif edu == "Master degree or equivalent":
        return f"Master's in {random.choice(UNIVERSITY_MAJORS)}"
    elif edu == "Doctoral degree or equivalent":
        return f"Ph.D. in {random.choice(UNIVERSITY_MAJORS)}"
    return edu


def create_batch_prompt(user_demographics: List[Dict]) -> str:
    """Build the batch generation prompt for a list of user demographics."""
    user_descriptions = [
        f"User {i}: {user['age']}-year-old {user['gender']}, {user['education']}"
        for i, user in enumerate(user_demographics, 1)
    ]

    prompt = f"""
    You are an expert in Digital Sociology and Internet Linguistics. Your task is to synthesize realistic social media identities for a diverse population.

    ### Simulation Goal
    The goal is to mimic the diverse naming patterns observed on real social media platforms. For each user description provided, infer their likely **"Digital Fingerprint"** based on their demographics (Age, Education, Background). Avoid generic or bot-like patterns.

    ### Generation Guidelines:

    1. **REALNAME (Display Name):**
    - Objective: Reflect the user's self-perception.
    - Instruction: Ensure a mix of formatting styles. Some users use full legal names (e.g., "Jonathan Smith"), some use casual nicknames (e.g., "Jonny"), some include credentials (e.g., "Jon Smith, PhD"), and younger users often use stylistic lowercase or emojis (e.g., "jon .").

    2. **USERNAME (Handle):**
    - Objective: Create a unique human-like handle, reflect the **era** in which they likely joined the internet. Keep the handle typical of Twitter.
    - Style: Keep it between 8-15 characters and realistic for Twitter.
    - Digital Literacy:
        - High literacy (often younger) = Abstract, aesthetic, short, or witty handles.
        - Low literacy (often older) = Formulaic, name+sequence, or rigid handles.
    - Ban any numbers that can be inferred from the profile (e.g., age).
    - Other numbers are allowed if they look natural for Twitter handles.

    ### Input Data:
    {chr(10).join(user_descriptions)}

    ### Output Format:
    CRITICAL: Return ONLY a valid JSON array.
    [
    {{"realname": "...", "username": "..."}},
    ...
    ]
    """
    return prompt


def generate_batch_with_gpt(user_demographics: List[Dict], max_retries: int = 3) -> List[Dict]:
    """Generate realname/username for a batch of users via the LLM, with retries and a fallback."""
    for attempt in range(max_retries):
        try:
            prompt = create_batch_prompt(user_demographics)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates user profiles. You MUST respond with valid JSON only, no markdown formatting, no explanations."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.8,
            )

            content = response.choices[0].message.content.strip()

            try:
                # Strip possible markdown code fences.
                content_cleaned = content.strip()
                if content_cleaned.startswith("```json"):
                    content_cleaned = content_cleaned[7:]
                    if content_cleaned.endswith("```"):
                        content_cleaned = content_cleaned[:-3]
                elif content_cleaned.startswith("```"):
                    lines = content_cleaned.split("\n")
                    if len(lines) > 2:
                        content_cleaned = "\n".join(lines[1:-1])

                content_cleaned = content_cleaned.strip()
                user_profiles = json.loads(content_cleaned)

                # Pad or truncate to the requested count.
                target_count = len(user_demographics)
                if len(user_profiles) < target_count:
                    print(f"Warning: got {len(user_profiles)} < {target_count}; padding remainder via fallback")
                    missing = target_count - len(user_profiles)
                    fallback = generate_batch_fallback(user_demographics[-missing:])
                    user_profiles.extend(
                        [{"realname": fb["realname"], "username": fb["username"]} for fb in fallback]
                    )
                elif len(user_profiles) > target_count:
                    print(f"Warning: got {len(user_profiles)} > {target_count}; truncating to target")
                    user_profiles = user_profiles[:target_count]

                # Merge demographics with generated fields.
                result = []
                for i, (demo, profile) in enumerate(zip(user_demographics, user_profiles)):
                    result.append({
                        "user_id": None,  # assigned later
                        "realname": profile.get("realname", f"User {i + 1}"),
                        "username": profile.get("username", f"user{i + 1}"),
                        "age": demo["age"],
                        "gender": demo["gender"],
                        "education": demo["education"],
                    })
                return result

            except json.JSONDecodeError as e:
                print(f"JSON parse failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Raw response: {content[:100]}...")

                # More aggressive salvage: extract the outermost JSON array.
                try:
                    start_idx = content.find("[")
                    end_idx = content.rfind("]")
                    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                        user_profiles = json.loads(content[start_idx:end_idx + 1])
                        print("Recovered JSON via aggressive cleanup")
                    else:
                        raise json.JSONDecodeError("no valid JSON array found", content, 0)
                except json.JSONDecodeError:
                    if attempt == max_retries - 1:
                        print("All JSON parse attempts failed; using fallback...")
                        return generate_batch_fallback(user_demographics)
                    print("JSON parse failed; retrying...")
                    time.sleep(2)
                    continue

        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return generate_batch_fallback(user_demographics)
            time.sleep(1)

    return generate_batch_fallback(user_demographics)


def generate_batch_fallback(user_demographics: List[Dict]) -> List[Dict]:
    """Template-based fallback generator used when the LLM call fails."""
    print("Using template fallback generator...")

    first_names_female = ["Emma", "Olivia", "Ava", "Sophia", "Isabella", "Charlotte",
                          "Amelia", "Mia", "Harper", "Evelyn", "Abigail", "Emily"]
    first_names_male = ["Liam", "Noah", "Oliver", "Elijah", "William", "James",
                        "Benjamin", "Lucas", "Henry", "Alexander", "Mason", "Michael"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                  "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez"]

    result = []
    for demo in user_demographics:
        first_name = random.choice(first_names_female if demo["gender"] == "female" else first_names_male)
        last_name = random.choice(last_names)

        if "Ph.D." in demo["education"]:
            realname = f"Dr. {first_name} {last_name}"
        elif random.random() < 0.3:
            middle_initial = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            realname = f"{first_name} {middle_initial}. {last_name}"
        else:
            realname = f"{first_name} {last_name}"

        if demo["age"] <= 25:
            username_styles = [
                f"{first_name.lower()}{random.randint(90, 99)}",
                f"{first_name.lower()}_{last_name.lower()}",
                f"real{first_name.lower()}",
                f"{first_name.lower()}{random.randint(2000, 2010)}",
            ]
        else:
            username_styles = [
                f"{first_name.lower()}{last_name.lower()}",
                f"{first_name[0].lower()}{last_name.lower()}",
                f"{first_name.lower()}{random.randint(70, 99)}",
            ]

        result.append({
            "user_id": None,
            "realname": realname,
            "username": random.choice(username_styles),
            "age": demo["age"],
            "gender": demo["gender"],
            "education": demo["education"],
        })

    return result


def generate_agents_batch(
    num_users: int = 6000,
    batch_size: int = 25,
    use_gpt: bool = True,
    balanced_gender: bool = True,
    max_workers: int = 5,
) -> List[Dict]:
    batch_count = (num_users + batch_size - 1) // batch_size
    print(f"=== Generating {num_users} agents (batch_size={batch_size}, use_gpt={use_gpt}) ===")
    print(f"Gender distribution: {'balanced (50/50)' if balanced_gender else 'empirical (36/64)'}")
    print(f"Estimated {batch_count} API call(s)")

    # Sample demographics for every user first.
    all_demographics = []
    for _ in range(num_users):
        age = generate_age()
        gender_probs = P_GENDERS_BALANCED if balanced_gender else P_GENDERS_REAL
        gender = weighted_random_choice(GENDERS, gender_probs)
        education = generate_education(age)
        all_demographics.append({"age": age, "gender": gender, "education": education})

    all_users = []

    if use_gpt and max_workers > 1:
        print(f"Processing batches with {max_workers} concurrent workers...")

        def process_batch(batch_idx):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_users)
            return generate_batch_with_gpt(all_demographics[start_idx:end_idx])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, i) for i in range(batch_count)]
            for i, future in enumerate(as_completed(futures)):
                all_users.extend(future.result())
                print(f"Completed batch {i + 1}/{batch_count}, total users: {len(all_users)}")
    else:
        print("Processing batches sequentially...")
        for batch_idx in range(batch_count):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_users)
            batch_demographics = all_demographics[start_idx:end_idx]
            print(f"Processing batch {batch_idx + 1}/{batch_count} (users {start_idx + 1}-{end_idx})...")

            if use_gpt:
                all_users.extend(generate_batch_with_gpt(batch_demographics))
            else:
                all_users.extend(generate_batch_fallback(batch_demographics))

            if use_gpt and batch_idx < batch_count - 1:
                time.sleep(0.1)  # gentle rate limiting

    # Assign sequential user ids.
    for i, user in enumerate(all_users):
        user["user_id"] = i

    print(f"Generated {len(all_users)} agents")
    return all_users


def save_agents_json(agents: List[Dict], filename: str):
    """Write agents to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(agents, f, ensure_ascii=False, indent=2)
    print(f"Saved agents to {filename}")


def save_agents_csv(agents: List[Dict], filename: str) -> pd.DataFrame:
    """Write agents to a CSV file in the OASIS-compatible schema."""
    csv_data = []
    for agent in agents:
        csv_data.append({
            "user_id": agent["user_id"],
            "username": agent["username"],
            "name": agent["realname"],
            "following_agentid_list": "[]",
            "previous_tweets": "[]",
            "user_char": "",
            "description": f"{agent['age']}-year-old {agent['gender']} with {agent['education']}",
            "age": agent["age"],
            "gender": agent["gender"],
            "education": agent["education"],
        })

    column_order = [
        "user_id", "username", "name", "following_agentid_list",
        "previous_tweets", "user_char", "description",
        "age", "gender", "education",
    ]
    df = pd.DataFrame(csv_data)[column_order]
    df.to_csv(filename, mode="w", header=True, index=False, encoding="utf-8")
    print(f"Saved agents to {filename} (OASIS-compatible format)")
    return df


if __name__ == "__main__":
    random.seed(42)

    NUM_USERS = 1000
    BATCH_SIZE = 5
    USE_GPT = True
    BALANCED_GENDER = False
    MAX_WORKERS = 5

    OUTPUT_DIR = "./batches/new"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    agents = generate_agents_batch(
        num_users=NUM_USERS,
        batch_size=BATCH_SIZE,
        use_gpt=USE_GPT,
        balanced_gender=BALANCED_GENDER,
        max_workers=MAX_WORKERS,
    )

    save_agents_json(agents, f"{OUTPUT_DIR}/agents_{NUM_USERS}.json")
    save_agents_csv(agents, f"{OUTPUT_DIR}/agents_{NUM_USERS}.csv")
