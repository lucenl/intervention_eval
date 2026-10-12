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
P_GENDERS = [0.363, 0.637]

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
    - Instruction: Ensure a mix of formatting styles. Some users use full legal names (e.g., "Jonathan Smith"), some use casual nicknames (e.g., "Jonny"), some include credentials (e.g., "Jon Smith, PhD"), and younger users often use stylistic lowercase or emojis (e.g., "jon ☾").

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
    """Generate realname/username for a batch of users via the LLM, with retries."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates user profiles. You MUST respond with valid JSON only, no markdown formatting, no explanations."},
                    {"role": "user", "content": create_batch_prompt(user_demographics)},
                ],
                max_tokens=2000,
                temperature=0.8,
            )
            content = response.choices[0].message.content.strip()

            # Strip possible markdown code fences.
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            user_profiles = json.loads(content)

            if len(user_profiles) < len(user_demographics):
                raise ValueError(f"got {len(user_profiles)} < {len(user_demographics)} profiles")
            user_profiles = user_profiles[:len(user_demographics)]

            # Merge demographics with generated fields.
            return [{
                "user_id": None,
                "realname": profile["realname"],
                "username": profile["username"],
                "age": demo["age"],
                "gender": demo["gender"],
                "education": demo["education"],
            } for demo, profile in zip(user_demographics, user_profiles)]

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(1)

    raise RuntimeError(f"Name generation failed after {max_retries} attempts")


def generate_agents_batch(num_users: int, batch_size: int = 25, max_workers: int = 5) -> List[Dict]:
    batch_count = (num_users + batch_size - 1) // batch_size
    print(f"=== Generating {num_users} agents in {batch_count} batches of {batch_size} ===")

    # Sample demographics for every user first.
    all_demographics = []
    for _ in range(num_users):
        age = generate_age()
        all_demographics.append({
            "age": age,
            "gender": weighted_random_choice(GENDERS, P_GENDERS),
            "education": generate_education(age),
        })

    def process_batch(batch_idx):
        start_idx = batch_idx * batch_size
        return generate_batch_with_gpt(all_demographics[start_idx:start_idx + batch_size])

    all_users = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, i) for i in range(batch_count)]
        for i, future in enumerate(as_completed(futures)):
            all_users.extend(future.result())
            print(f"Completed batch {i + 1}/{batch_count}, total users: {len(all_users)}")

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
    df = pd.DataFrame([{
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
    } for agent in agents])
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Saved agents to {filename} (OASIS-compatible format)")
    return df
