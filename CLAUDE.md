# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

DietAI24 is a research pipeline for AI-based dietary assessment from food images. Given a photo of food, it uses a GPT-4o vision + RAG pipeline to (1) infer an eight-digit FNDDS food code, (2) estimate portion size, and (3) look up/compute nutrient amounts. It's evaluated against the ASA24 and Nutrition5k public food-image datasets, with several third-party baselines (Foodvisor, CalorieMama, plain ChatGPT, a ViT+ElasticNet model) for comparison.

All Python source lives in `code/`; scripts are written to be run with `code/` as the working directory and use `../`-relative paths to reach data directories.

## Environment setup

```bash
conda env create -f env/env.yml   # creates conda env named "llm"
conda activate llm
```

- `code/.env` holds `OPENAI_API_KEY` (and is git-ignored). Every OpenAI-calling script does `dotenv.load_dotenv()`/`load_dotenv()` at import/startup — never hardcode keys in source.
- There is no requirements.txt, lint config, or test suite in this repo. Scripts are invoked directly with `python <script>.py [args]` from inside `code/`.
- Model names (chat, vision, embedding) are centralized in `code/config.py` (`MODELS` dict) — update there rather than hardcoding model strings in individual scripts.

## Data layout (repo-root, sibling to `code/`)

These directories hold large downloaded/derived datasets and are mostly untracked in git (see `.gitignore`); don't assume they exist in a fresh checkout — they must be fetched/generated per the dataset instructions in `README.md`:

- `ASA24/` — raw ASA24 image + metadata (from the ASA24 portion-size dataset)
- `Nutrition5k/` — raw Nutrition5k metadata/images
- `FNDDS/` — USDA FNDDS food-code and nutrient-value reference tables (xlsx/csv/mdb)
- `NHANES/` — NHANES dietary recall data, used to build a top-1000 frequent-foods reference
- Root-level CSVs (`df_image_link.csv`, `df_results.csv`, `output.csv`, `ASA24_GPTFoodCodes_*.csv`, `dish_metadata.csv`) are pipeline intermediates/outputs produced by the scripts below, consumed by later stages.

## Pipeline architecture

The scripts form a sequential pipeline; each stage reads CSVs written by the previous one. Run everything from `code/`.

**1. Data preparation** (build reference/input CSVs from raw datasets)
- `asa24_metadata_proc.py` — reads ASA24 image metadata (xlsx), builds `df_image_link.csv` (all portion-image references) and `df_results.csv`/`ASA24_GPTFoodCodes_portion.csv` (sampled test rows), optionally filtered to food codes that have FNDDS portion-weight data.
- `nutrition5k_proc.py` — parses Nutrition5k's headerless dish-metadata CSVs into a clean ingredients table, dedupes by ingredient set, and filters to dishes with reachable images.
- `fndds_proc.py` — converts the FNDDS "Foods and Beverages" xlsx into a CSV with a natural-language `Food description` (used as RAG document text).
- `asa_proc.py` — builds the NHANES top-1000-frequent-foods report used to scope/prioritize food codes.

**2. Food code inference** (`rag_food_code.py`)
- Loads the FNDDS description CSV into a Chroma vector store (OpenAI embeddings), wraps it in a `MultiQueryRetriever` (LLM rewrites the query 5 ways to improve recall).
- For each image: GPT-4o vision describes the food (`get_messages_from_url`), then the description is used to retrieve candidate FNDDS entries and a chat prompt selects/returns the 8-digit food code.
- Long-running and checkpointed: `--checkpoint_file` stores `num,index` so a killed/rate-limited run resumes where it left off; results are written to `--results_file` after every row (crash-safe, not batched).
- Retries on rate limits (`429`) with exponential backoff; non-rate-limit failures are captured into the output row rather than raised.

**3. Portion size estimation** (`rag_portion_size.py`)
- Reuses the `Vision` class from `chagApp_openai.py` (plain OpenAI vision chat, stateful message history per instance — not the LangChain stack used in step 2).
- Two prompt modes: `shot` (pick the closest portion-size option from a multiple-choice list built from `PortionShot`) and `amount` (estimate a numeric quantity in the given unit). Both are run per row against `ASA24_GPTFoodCodes_portion.csv`.
- Writes back to the same `ASA24_GPTFoodCodes_portion.csv` after every row.

**4. Nutrient estimation** (two variants, dataset-dependent)
- `nutrient_estimate.py` — ASA24 path: joins `ASA24_GPTFoodCodes_portion.csv` against `FNDDS/FoodWeights.csv` (food code + portion code → weight in grams) and `FNDDS/...FNDDS Nutrient Values.xlsx`, scales per-100g nutrient values by portion weight.
- `nutrient_estimate_mix.py` — Nutrition5k path: parses free-text GPT ingredient/weight strings (`GPTAmount` column), matches ingredient names to FNDDS food codes, and sums weighted nutrient contributions per dish.

**5. Reporting** (`nutrition_report.py`)
- Reads `ASA24_GPTFoodCodes_nutrition.csv`, inlines thumbnail images as base64 data URIs, and renders a single self-contained HTML report (`ASA24_GPTFoodCodes_nutrition_report.html`) with a summary (avg weight/energy/protein) and a sortable table of `CORE_COLUMNS`.

## Baselines (`code/baseline_*.py`, `code/ViT/`)

Independent of the main pipeline, used only for comparison:
- `baseline_foodvisor.py`, `baseline_calorieMaMa.py` — call third-party food-recognition APIs directly (hardcoded/empty API keys/paths — treat as templates, not runnable as-is).
- `baseline_chatGPT.py` — per-nutrient direct-estimation prompting via `chagApp_openai.Vision`, no RAG/food-code step.
- `ViT/` — a separate ViT-embedding + multi-task ElasticNet baseline (`exec_ENet.py`, `dataset_utils.py`) with its own SLURM job script (`run.sh`); trains/evaluates on Nutrition5k and ASA24 dish datasets, independent of the OpenAI-based pipeline above.

## Two OpenAI client wrappers — don't confuse them

- `chagApp_openai.py` — `Vision` class using the public OpenAI SDK (`OpenAI()`, reads `OPENAI_API_KEY` from env via dotenv). This is what the current pipeline (`rag_portion_size.py`, `baseline_chatGPT.py`) actually uses.
- `chatApp_azure.py` — `ChatApp`/`Vision` classes for Azure OpenAI, with empty `api_key`/`api_base` placeholders. Not wired into the current pipeline; treat as a template if Azure support is needed.

## Conventions to follow when editing these scripts

- Preserve the incremental-write/checkpoint pattern in long-running, paid-API-calling loops (`rag_food_code.py`, `rag_portion_size.py`, baselines) — write results to CSV after each row/item and catch-and-record per-item exceptions rather than letting one bad row abort the whole run.
- Scripts assume they're launched with CWD = `code/`; file paths are relative (`../FNDDS/...`, `../ASA24/...`). Keep new scripts consistent with this rather than switching to absolute paths.
- FNDDS food codes are zero-padded 8-digit strings (`f"{int(value):08d}"`); compare/join on the string form, not int, to avoid dropping leading zeros.
