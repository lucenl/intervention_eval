#!/usr/bin/env python3
import argparse
import asyncio
import csv
import os
import random
import shutil
from datetime import datetime
from typing import Dict

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, OpenAIBackendRole
from yaml import safe_load

from oasis import ActionType, LLMAction, generate_twitter_agent_graph
from oasis.social_platform.typing import RecsysType

from api_limits import make_env, track_usage
from unified_treatment import UnifiedExperimentTreatment

KEEP_STEPS = 5


# ============ Memory Truncation ============


def truncate_agent_memory(agent, keep_steps=KEEP_STEPS):
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
        if chat_records[i].memory_record.role_at_backend == OpenAIBackendRole.USER:
            found_user_msgs += 1
            if found_user_msgs == keep_steps:
                cut_index = i
                break

    if cut_index == 0:
        return len(all_records)

    agent.memory.clear()
    for r in system_records + chat_records[cut_index:]:
        agent.memory.write_record(r.memory_record)
    return len(agent.memory.retrieve())


# ============ Main Simulation Loop ============
async def main(config: Dict, args: argparse.Namespace):
    model_cfg = config['model']
    data_config = config['data']
    sim_cfg = config['simulation']
    twitter_cfg = config['twitter']
    api_cfg = config['api']
    prices = (api_cfg.get('price_per_1m_input'), api_cfg.get('price_per_1m_output'))
    track_cost = None not in prices

    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    token_log_dir = "./token_logs"
    os.makedirs(token_log_dir, exist_ok=True)
    token_csv_path = os.path.join(token_log_dir, f"{args.name}_{run_stamp}_tokens.csv")
    with open(token_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["Step", "Active_Agents", "Avg_Memory_Len",
                                "Total_Input", "Total_Output", "Cost"])

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType(model_cfg['type']),
        model_config_dict={"temperature": model_cfg['temperature']},
    )
    usage = track_usage(model)
    available_actions = [ActionType.LIKE_POST, ActionType.REPOST,
                         ActionType.FOLLOW, ActionType.CREATE_COMMENT]

    csv_path = args.agents_file
    print(f"Loading agents from: {csv_path}")
    agent_graph = await generate_twitter_agent_graph(
        profile_path=csv_path,
        model=model,
        available_actions=available_actions,
    )
    total_agents = len(agent_graph.get_agents())
    print(f"Loaded {total_agents} agents")

    base_db_path = data_config['db_path']
    output_db_dir = args.output_dir or os.path.dirname(base_db_path)
    os.makedirs(output_db_dir, exist_ok=True)
    db_path = os.path.join(output_db_dir,
                           f"{args.name}_{run_stamp}.db")
    source_db = args.initial_db or base_db_path
    print(f"Copying DB from [{source_db}] to [{db_path}]...")
    shutil.copy2(source_db, db_path)

    env = make_env(
        agent_graph=agent_graph,
        database_path=db_path,
        max_concurrent=api_cfg['max_concurrent'],
        requests_per_minute=api_cfg['requests_per_minute'],
    )
    env.platform.refresh_rec_post_count = twitter_cfg['refresh_rec_post_count']
    env.platform.max_rec_post_len = twitter_cfg['max_rec_post_len']
    env.platform.political_ratio = twitter_cfg['political_ratio']
    env.platform.recsys_type = RecsysType.RANDOM

    with open(csv_path, newline="", encoding="utf-8") as f:
        global_id_offset = int(next(csv.DictReader(f))["user_id"])
    env.platform.treatment = UnifiedExperimentTreatment(
        celebrities=config['celebrities'], global_id_offset=global_id_offset)

    await env.reset()

    num_timesteps = args.num_timesteps or sim_cfg['num_timesteps']
    activation_prob = sim_cfg['activate_prob']
    print(f"\nRunning {num_timesteps} steps (memory window {KEEP_STEPS})")

    for step in range(num_timesteps):
        step_number = step + 1
        print(f"\n=== Step {step_number}/{num_timesteps} ===")
        actions = {}
        active_users = []
        for agent_id in range(total_agents):
            if random.random() < activation_prob:
                actions[env.agent_graph.get_agent(agent_id)] = LLMAction()
                active_users.append(agent_id)

        if actions:
            print(f"Executing actions for {len(actions)} agents...")
            await env.step(actions, step_number=step_number)
            await env.platform.record_rec_history(step_number)

        mem_lens = [truncate_agent_memory(env.agent_graph.get_agent(uid))
                    for uid in active_users]

        avg_mem = sum(mem_lens) / len(mem_lens) if mem_lens else 0
        cost = f"{usage.cost(*prices):.4f}" if track_cost else ""
        with open(token_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([step_number, len(active_users), f"{avg_mem:.1f}",
                                    usage.prompt_tokens, usage.completion_tokens, cost])

    env.platform.treatment.save_assignments_to_db(db_path)
    print("\nSimulation completed!")
    await env.close()
    return db_path


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="unified_config.yaml")
    parser.add_argument("--agents_file", type=str, required=True,
                        help="Agent CSV for this batch")
    parser.add_argument("--name", type=str, default="exp_unified",
                        help="Prefix of the output DB file name")
    parser.add_argument("--num_timesteps", type=int, default=None,
                        help="Override simulation.num_timesteps")
    parser.add_argument("--initial_db", type=str, default=None,
                        help="Path to a previous DB to resume from")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the output DB")
    return parser.parse_args()


def main_entry():
    args = setup_args()
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = safe_load(f)
    asyncio.run(main(config, args))
    return 0


if __name__ == "__main__":
    exit(main_entry())
