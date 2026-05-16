# OASIS Simulation Pipeline

Pipeline for the OASIS simulation across five treatments
(Control / Inject-2 / Inject-4 / Upranking / Celebrity).

## Layout

```
pipeline_bundle/
├── load_posts_with_users.py     # build preloaded_data.db
├── run_pipeline.py              # orchestrator
├── addnews_simulation.py        # per-batch simulator
├── unified_treatment.py         # 5-condition treatment controller
├── treatment_base.py            # abstract Treatment base
├── API_manager.py               # OASIS env helpers
├── unified_config.yaml          # simulation config (incl. celebrity list)
├── agents/gen.py                # LLM agent generator
├── data/                        # content pool + users (distributed separately)
└── database/                    # holds preloaded_data.db
```

## Prerequisites

- Python ≥ 3.10
- `pip install camel-ai pandas pyyaml openai` + the OASIS package
- `OPENAI_API_KEY` exported in the environment
- Place `all_posts_final.parquet` and `final_users.csv` into `data/`

## Run

```bash
python load_posts_with_users.py     # build preloaded_data.db from data/
python run_pipeline.py              # run the simulation
```
