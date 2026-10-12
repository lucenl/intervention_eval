# OASIS Simulation Pipeline

Pipeline for the OASIS simulation across five treatments (Control / Inject-2 / Inject-4 / Upranking / Celebrity).

## Layout

```
├── load_posts_with_users.py     # builds the starting database from the posts and users in data/
├── run_pipeline.py              # main entry point: creates agents and runs all batches of experiments 
├── run_simulation.py            # runs one batch of agents for all time steps (called by run_pipeline.py)
├── unified_treatment.py         # assigns each agent a condition and changes its feed accordingly
├── treatment_base.py            # base class for unified_treatment.py
├── api_limits.py                # limits the rate of OpenAI requests and counts tokens used
├── unified_config.yaml          # settings: model, number of steps, feed length, celebrity list, API
├── agents/gen.py                # creates agent profiles (age, gender, education, name, username)
├── oasis/                       # the social media platform the agents use (modified OASIS)
├── data/                        # input posts and users go here (not included)
└── database/                    # the starting database is written here
```

## Prerequisites

* Python ≥ 3.10
* `pip install -r requirements.txt` (OASIS is included in `oasis/`)
* `OPENAI_API_KEY` exported in the environment
* Place `all_posts_final.parquet` and `final_users.csv` into `data/`

## Run

```
python load_posts_with_users.py     # build preloaded_data.db from data/
python run_pipeline.py              # run the simulation
```

## OASIS

`oasis/` is a modified copy of [camel-ai/oasis](https://github.com/camel-ai/oasis) (Apache-2.0, see `LICENSE_OASIS`).
