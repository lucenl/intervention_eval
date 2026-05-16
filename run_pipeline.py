#!/usr/bin/env python3
import os
import time
import yaml
import json
import glob
import asyncio
import subprocess
from datetime import datetime

try:
    from agents.gen import generate_agents_batch, save_agents_csv, save_agents_json
except ImportError:
    print("Error: Could not import 'gen.py'. Please make sure it is in the same directory.")
    exit(1)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TOTAL_AGENTS = 16000
BATCH_SIZE = 1000
START_BATCH_IDX = 14

BASE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "unified_config.yaml")
DATA_DIR = os.path.join(PROJECT_ROOT, "pipeline_data")

AGENTS_DIR = os.path.join(DATA_DIR, "agents")
CONFIGS_DIR = os.path.join(DATA_DIR, "configs")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
DB_OUTPUT_DIR = os.path.join(DATA_DIR, "databases")

MAX_CONCURRENT_BATCHES = 4
COOLDOWN_AFTER_CHUNK = 30

for d in [AGENTS_DIR, CONFIGS_DIR, LOGS_DIR, DB_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)


def load_existing_usernames(agents_dir):
    seen_usernames = set()
    files = glob.glob(os.path.join(agents_dir, "*.json"))
    print(f"Scanning {len(files)} JSON files for usernames...")

    for idx, f_path in enumerate(files, 1):
        if idx % 500 == 0:
            print(f"  Processed {idx}/{len(files)} files...")
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for agent in data:
                    uname = agent.get('username', '').strip().lower()
                    if uname:
                        seen_usernames.add(uname)
        except Exception:
            pass

    print(f"Loaded {len(seen_usernames)} unique usernames.")
    return seen_usernames


async def run_single_simulation(batch_idx: int, batch_config_path: str, db_output_dir: str):
    batch_id_str = f"batch_{batch_idx}"
    run_log_path = os.path.join(LOGS_DIR, f"run_{batch_id_str}.log")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting simulation for Batch {batch_idx + 1}")

    cmd = [
        "python3", "-u", "addnews_simulation.py",
        "--config_path", batch_config_path,
        "--output_dir", db_output_dir
    ]

    try:
        with open(run_log_path, "w") as log_file:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            await process.wait()

        if process.returncode == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch {batch_idx + 1} completed.")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch {batch_idx + 1} failed (code {process.returncode}). Check {run_log_path}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Execution error for batch {batch_idx + 1}: {e}")


async def run_all_batches_parallel():
    num_batches = (TOTAL_AGENTS + BATCH_SIZE - 1) // BATCH_SIZE

    if START_BATCH_IDX >= num_batches:
        print(f"START_BATCH_IDX={START_BATCH_IDX} >= total batches={num_batches}. Nothing to run.")
        return

    global_seen_usernames = load_existing_usernames(AGENTS_DIR)

    print(f"\nStarting Parallel Pipeline")
    print(f"   Goal: {TOTAL_AGENTS} agents | Batches: {START_BATCH_IDX} -> {num_batches-1} | Concurrent: {MAX_CONCURRENT_BATCHES}")

    pending_tasks = []

    for batch_idx in range(START_BATCH_IDX, num_batches):
        print(f"\n{'_'*60}")
        print(f"Processing Batch {batch_idx + 1}/{num_batches} (idx={batch_idx})")

        batch_id_str = f"batch_{batch_idx}"
        agent_csv_path = os.path.join(AGENTS_DIR, f"agents_{batch_id_str}.csv")
        agent_json_path = os.path.join(AGENTS_DIR, f"agents_{batch_id_str}.json")
        batch_config_path = os.path.join(CONFIGS_DIR, f"config_{batch_id_str}.yaml")

        # Step 1: Generate Agents
        if os.path.exists(agent_csv_path) and os.path.exists(agent_json_path):
            print(f"Agents exist. Skipping generation.")
        else:
            print(f"Generating {BATCH_SIZE} agents...")
            try:
                agents = generate_agents_batch(
                    num_users=BATCH_SIZE,
                    batch_size=25,
                    use_gpt=True,
                    balanced_gender=False,
                    max_workers=5
                )
                global_start_id = batch_idx * BATCH_SIZE
                print(f"Assigning IDs & de-duping...")
                duplicates_fixed = 0
                for i, agent in enumerate(agents):
                    real_id = global_start_id + i
                    agent['user_id'] = real_id
                    original_name = agent.get('username') or f"user_{real_id}"
                    normalized = original_name.strip().lower()
                    if normalized in global_seen_usernames:
                        agent['username'] = f"{original_name}_{real_id}"
                        global_seen_usernames.add(agent['username'].lower())
                        duplicates_fixed += 1
                    else:
                        global_seen_usernames.add(normalized)
                if duplicates_fixed:
                    print(f"Fixed {duplicates_fixed} duplicates.")
                save_agents_json(agents, agent_json_path)
                save_agents_csv(agents, agent_csv_path)
                print(f"Agents saved at {datetime.now().strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"Generation failed for batch {batch_idx}: {e}")
                break

        # Step 2: Prepare Config
        try:
            with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            config_data['data']['agents_file'] = agent_csv_path
            config_data['experiment']['name'] = f"exp_unified_{batch_id_str}"
            with open(batch_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            print(f"Config ready.")
        except Exception as e:
            print(f"Config prep failed: {e}")
            break

        # Step 3: Queue simulation
        print(f"Queuing simulation task...")
        pending_tasks.append(
            run_single_simulation(batch_idx, batch_config_path, DB_OUTPUT_DIR)
        )

        if len(pending_tasks) >= MAX_CONCURRENT_BATCHES:
            print(f"Launching {len(pending_tasks)} parallel simulations...")
            await asyncio.gather(*pending_tasks)
            pending_tasks = []
            print(f"Chunk done -> cooling down {COOLDOWN_AFTER_CHUNK}s...")
            await asyncio.sleep(COOLDOWN_AFTER_CHUNK)

    if pending_tasks:
        print(f"Launching final {len(pending_tasks)} parallel simulations...")
        await asyncio.gather(*pending_tasks)

    print(f"\nAll batches completed! Results in {DB_OUTPUT_DIR}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set.")
    asyncio.run(run_all_batches_parallel())
