#!/usr/bin/env python3
import asyncio
import argparse
import os
import json
import random
import logging
import csv
from datetime import datetime
from time import sleep
from typing import Dict, Optional
import shutil
from collections import Counter

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, OpenAIBackendRole
from yaml import safe_load
from oasis.social_platform.typing import RecsysType

from unified_treatment import UnifiedExperimentTreatment

import oasis
from oasis import (ActionType, LLMAction, ManualAction, generate_twitter_agent_graph)


from oasis.social_platform.config.user import UserInfo
from oasis.social_agent.agent_environment import SocialEnvironment
from string import Template

EXPERIMENT_CONFIG = {}


# ============ Per-Agent Logging ============


def log_agent_memory_state(agent, agent_id, step, phase, base_dir="./agent_specific_logs"):
    """Dump an agent's current memory to a per-agent log file."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    filename = os.path.join(base_dir, f"agent_{agent_id}_memory.log")
    records = agent.memory.retrieve()

    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"TIME STEP: {step} | PHASE: {phase}\n")
        f.write(f"{'='*80}\n")

        for i, r in enumerate(records):
            msg_obj = r.memory_record.message
            role = str(r.memory_record.role_at_backend).replace("OpenAIBackendRole.", "")

            display_content = ""

            if role == "ASSISTANT":
                if hasattr(msg_obj, 'func_name') and msg_obj.func_name:
                    args_str = str(msg_obj.args) if hasattr(msg_obj, 'args') else "{}"
                    display_content = f"CALL: {msg_obj.func_name}({args_str})"
                elif not msg_obj.content and hasattr(msg_obj, 'tool_calls'):
                    calls = [f"{tc.function.name}({tc.function.arguments})" for tc in msg_obj.tool_calls]
                    display_content = f"CALLS: {', '.join(calls)}"
                else:
                    display_content = msg_obj.content

            elif role == "FUNCTION":
                if hasattr(msg_obj, 'result'):
                    display_content = f"RESULT: {msg_obj.result}"
                else:
                    display_content = msg_obj.content

            elif role == "USER":
                raw = msg_obj.content
                if "social media feed" in raw:
                    snippet = raw[:100].replace('\n', ' ')
                    display_content = f"FEED: {snippet}..."
                else:
                    display_content = raw[:200]

            elif role == "SYSTEM":
                display_content = "SYSTEM PROMPT (Hidden for brevity)"

            else:
                display_content = str(msg_obj.content)[:200]

            clean_display = str(display_content).replace('\n', ' ')

            f.write(f"Msg {i:<2} | {role:<10} | {clean_display}\n")

        f.write(f"\nSUMMARY: Agent {agent_id} has {len(records)} messages.\n")
        f.write(f"{'-'*80}\n")


# ============ Memory Truncation ============


def truncate_agent_memory(agent, keep_steps=5):
    """Keep the system prompt plus the most recent keep_steps turns."""
    all_records = agent.memory.retrieve()
    if not all_records:
        return 0

    system_records = []
    chat_records = []

    for r in all_records:
        if r.memory_record.role_at_backend == OpenAIBackendRole.SYSTEM:
            system_records.append(r)
        else:
            chat_records.append(r)

    cut_index = 0
    found_user_msgs = 0

    for i in range(len(chat_records) - 1, -1, -1):
        role = chat_records[i].memory_record.role_at_backend
        if role == OpenAIBackendRole.USER:
            found_user_msgs += 1
            if found_user_msgs == keep_steps:
                cut_index = i
                break

    if cut_index == 0:
        return len(all_records)

    kept_chat_records = chat_records[cut_index:]

    agent.memory.clear()
    for r in system_records:
        agent.memory.write_record(r.memory_record)
    for r in kept_chat_records:
        agent.memory.write_record(r.memory_record)

    return len(agent.memory.retrieve())


def custom_twitter_system_message(self) -> str:
    name_string = ""
    if self.name is not None:
        name_string = f"Your name is {self.name}."
    if self.user_name is not None:
        name_string += f"\nYour username is {self.user_name}."

    if self.profile and "other_info" in self.profile:
        other_info = self.profile["other_info"]
        user_profile = self.profile["other_info"]["user_profile"]

        description = f"{name_string}"
        description += (
            f"\nYou are a {str(other_info.get('age', 'unknown'))} years old "
            f"{other_info.get('gender', 'person')}. Your education background is "
            f"{other_info.get('education', 'unknown')}."
        )

        if EXPERIMENT_CONFIG.get('include_bio', False) and user_profile:
            if str(user_profile).lower() != 'nan' and len(str(user_profile)) > 2:
                description += f"\nYour have profile:\n{user_profile}."
    else:
        description = name_string

    objective = "You're a Twitter user, and I'll present you with some posts. After you see the posts, choose some actions from the following functions."

    if EXPERIMENT_CONFIG.get('include_celebrity_hint', False):
        objective += " You may see posts from famous celebrities marked with (☑️ Verified)."

    system_content = f"""
# OBJECTIVE
{objective}

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.
{description}

# RESPONSE METHOD
Please perform actions by tool calling.

Rules you MUST follow EVERY TIME:
1. You MUST call at least one tool (like_post, comment_post, repost, follow).
2. You are NOT a passive observer. You are NOT allowed to just summarize and say "let me know what to do".
3. Do NOT write long summaries, lists, or prose unless it is inside the short "reason" field (max 2 sentences).
4. Your response must trigger tool use when appropriate — this is how you take action.
    """
    return system_content


def apply_experiment_patch(experiment_config: dict):
    global EXPERIMENT_CONFIG
    EXPERIMENT_CONFIG = experiment_config
    UserInfo.to_twitter_system_message = custom_twitter_system_message

    if EXPERIMENT_CONFIG.get('new_prompt', False):
        base_template_str = "$posts_env\nRead through those posts..."
    else:
        base_template_str = SocialEnvironment.env_template.template

    constraint_prompt = ""
    if EXPERIMENT_CONFIG.get('feed_constraint_prompt', False):
        constraint_prompt = (
            " You are strictly FORBIDDEN from interacting with any post_id that is not explicitly listed in the current feed provided to you.")
        base_template_str += constraint_prompt

    SocialEnvironment.env_template = Template(base_template_str)
    print(f"Applied experiment patch: {experiment_config.get('name', 'unknown')}")


# ============ Main Simulation Loop ============
async def main(config: Dict, args: argparse.Namespace):
    print("Starting Simulation with Per-Agent Full Logging")

    VERIFY_LOG_DIR = "./verification_logs"
    AGENT_LOG_DIR = "./agent_specific_logs"

    if os.path.exists(AGENT_LOG_DIR):
        shutil.rmtree(AGENT_LOG_DIR)
    os.makedirs(AGENT_LOG_DIR, exist_ok=True)
    if not os.path.exists(VERIFY_LOG_DIR):
        os.makedirs(VERIFY_LOG_DIR)

    TOKEN_CSV_PATH = os.path.join(VERIFY_LOG_DIR, "token_usage_trace.csv")
    with open(TOKEN_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Active_Agents", "Avg_Memory_Len", "Total_Input", "Total_Output", "Cost"])

    try:
        experiment_config = config.get('experiment', {})
        apply_experiment_patch(experiment_config)

        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0.7},
        )

        available_actions = [ActionType.LIKE_POST, ActionType.REPOST, ActionType.FOLLOW, ActionType.CREATE_COMMENT]

        data_config = config.get('data', {})
        csv_path = data_config.get('agents_file', 'us_twitter_users_2024.csv')
        print(f"Loading agents from: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        agent_graph = await generate_twitter_agent_graph(
            profile_path=csv_path,
            model=model,
            available_actions=available_actions,
        )
        total_agents = len(agent_graph.get_agents())
        print(f"Loaded {total_agents} agents")

        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_config.get('name', 'exp')
        base_db_path = data_config.get('db_path', './database/preloaded_data.db')
        base_db_dir, _ = os.path.split(base_db_path)
        db_path = os.path.join(base_db_dir, f"{exp_name}_{current_timestamp}.db")

        if os.path.exists(db_path):
            os.remove(db_path)
        shutil.copy2(base_db_path, db_path)

        print("Creating environment...")
        twitter_cfg = config.get('twitter', {})

        if args.output_dir:
            output_db_dir = args.output_dir
        else:
            output_db_dir = os.path.dirname(base_db_path)

        if not os.path.exists(output_db_dir):
            os.makedirs(output_db_dir, exist_ok=True)

        target_db_path = os.path.join(output_db_dir, f"{exp_name}_{current_timestamp}.db")

        source_db = base_db_path
        if args.initial_db:
            if os.path.exists(args.initial_db):
                print(f"[INCREMENTAL MODE] Resuming from previous DB: {args.initial_db}")
                source_db = args.initial_db
            else:
                print(f"Warning: Provided initial_db not found: {args.initial_db}. Using base DB.")

        print(f"Copying DB from [{source_db}] to [{target_db_path}]...")
        if os.path.exists(target_db_path):
            os.remove(target_db_path)
        shutil.copy2(source_db, target_db_path)

        from API_manager import create_chatagent_enhanced_oasis_env
        env = create_chatagent_enhanced_oasis_env(
            agent_graph=agent_graph,
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=target_db_path
        )

        env.platform.refresh_rec_post_count = twitter_cfg.get('refresh_rec_post_count', 40)
        env.platform.max_rec_post_len = twitter_cfg.get('max_rec_post_len', 40)
        env.platform.following_post_count = twitter_cfg.get('following_post_count', 10)
        env.platform.political_ratio = twitter_cfg.get('political_ratio', 0.10)
        celebs = (
            twitter_cfg.get('celebrities')
            or config.get('celebrities')
            or experiment_config.get('celebrities')
            or []
        )

        print("[Config] Setting RecsysType to RANDOM")
        env.platform.recsys_type = RecsysType.RANDOM

        print("Initializing Unified Experiment Controller...")
        unified_treatment = UnifiedExperimentTreatment(celebrities=celebs)
        env.platform.treatment = unified_treatment

        print("Resetting environment...")
        await env.reset()

        sim_cfg = config.get('simulation', {})
        num_timesteps = sim_cfg.get('num_timesteps', 3)
        activation_prob = sim_cfg.get('activate_prob', 0.1)
        KEEP_STEPS = 5

        print(f"\nRunning {num_timesteps} steps (Keep {KEEP_STEPS}). logs -> {AGENT_LOG_DIR}")

        for step in range(num_timesteps):
            step_number = step + 1
            print(f"\n=== Step {step_number}/{num_timesteps} ===")
            actions = {}
            active_users = []

            for agent_id in range(total_agents):
                if random.random() < activation_prob:
                    agent = env.agent_graph.get_agent(agent_id)
                    actions[agent] = LLMAction()
                    active_users.append(agent_id)

            if actions:
                print(f"Executing actions for {len(actions)} agents...")
                await env.step(actions, step_number=step_number)
                await env.platform.record_rec_history(step_number)
            else:
                print(f"No active agents in step {step_number}")

            current_mem_lens = []

            if active_users:
                print(f"processing memory for {len(active_users)} active agents...")

                for uid in active_users:
                    agent_obj = env.agent_graph.get_agent(uid)

                    new_len = truncate_agent_memory(agent_obj, keep_steps=KEEP_STEPS)
                    current_mem_lens.append(new_len)

            current_p = 0
            current_c = 0
            if hasattr(env, 'llm_semaphore'):
                current_p = env.llm_semaphore.total_prompt_tokens
                current_c = env.llm_semaphore.total_completion_tokens

            avg_mem = sum(current_mem_lens) / len(current_mem_lens) if current_mem_lens else 0
            est_cost = (current_p / 1e6 * 0.15) + (current_c / 1e6 * 0.60)

            with open(TOKEN_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([step_number, len(active_users),
                                f"{avg_mem:.1f}", current_p, current_c, f"{est_cost:.4f}"])

            if hasattr(env, 'llm_semaphore'):
                api_logger = logging.getLogger("chatagent_enhanced_semaphore")
                usage_msg = f"[Step {step_number} Total Usage] Input: {current_p} | Output: {current_c} | AvgMem: {avg_mem:.1f}"
                api_logger.info(usage_msg)

        env.platform.treatment.save_assignments_to_db(db_path)
        print("\nSimulation completed!")
        await env.close()
        return db_path

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--initial_db", type=str, default=None, help="Path to previous DB to resume from")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the output DB")
    return parser.parse_args()


def main_entry():
    import logging
    api_logger = logging.getLogger("chatagent_enhanced_semaphore")
    api_logger.setLevel(logging.WARNING)
    tracker_logger = logging.getLogger("chatagent_response_tracker")
    tracker_logger.setLevel(logging.WARNING)

    if not api_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        api_logger.addHandler(console_handler)

        log_dir = "./log"
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(os.path.join(log_dir, f"api_cost_{current_time}.log"), encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        api_logger.addHandler(file_handler)

    try:
        args = setup_args()
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = safe_load(f)
        asyncio.run(main(config, args))
        return 0
    except Exception as e:
        print(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main_entry())
