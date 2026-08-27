"""Benchmark the local vLLM pipeline's food-code/portion/nutrient accuracy against ASA24
ground truth.

Reuses analyze_image.py's single-code inference (infer_food_code, choose_portion,
compute_nutrients) rather than re-implementing the pipeline, so this measures exactly what
`python analyze_image.py --image ...` (without --multi-ingredient) would predict.

Ground-truth input is a CSV with the same columns as df_results.csv / df_results_clean10.csv:
FoodCode, FC_Description, Image, Link, Portion, PortionCode, PortionSubCode, Multiplier.
The true portion weight is looked up the same way nutrient_estimate.py does it:
FoodWeights.csv[(Food code, Portion code)]["Portion weight"] * Multiplier.

Usage:
    # Reuse the frozen 10-row sample already in the repo:
    python eval_asa24.py --sample_csv ../df_results_clean10.csv --output ../eval_asa24_clean10.csv

    # Draw a fresh random sample of N rows from df_results.csv instead:
    python eval_asa24.py --source_csv ../df_results.csv --n 10 --seed 0 --output ../eval_asa24_new10.csv
"""

import argparse

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from config import MODELS, LOCAL_VLLM_BASE_URL, LOCAL_VLLM_EMBED_BASE_URL
from rag_food_code import configure_chat_openai
from analyze_image import (
    DEFAULT_FNDDS_DESC_CSV,
    DEFAULT_NUTRIENTS_XLSX,
    DEFAULT_PERSIST_DIR,
    DEFAULT_PORTIONS_XLSX,
    REPORT_NUTRIENTS,
    build_vectordb,
    choose_portion,
    compute_nutrients,
    infer_food_code,
    load_portion_candidates,
)

DEFAULT_FOOD_WEIGHTS_CSV = "../FNDDS/FoodWeights.csv"


def true_portion_weight(food_code, portion_code, portion_subcode, multiplier, food_weights_df):
    """Ground-truth weight for a df_results row: FoodWeights.csv gives the weight of the FULL
    unit the ASA24 portion label refers to (e.g. "1 cup"); Multiplier scales that down to the
    actual reported portion (e.g. "1/4 cup" -> Multiplier 0.25)."""
    matches = food_weights_df[
        (food_weights_df["Food code"] == int(food_code)) & (food_weights_df["Portion code"] == int(portion_code))
    ]
    if "Subcode" in food_weights_df.columns and len(matches) > 1:
        matches = matches[matches["Subcode"] == portion_subcode]
    if matches.empty:
        raise RuntimeError(f"No FoodWeights.csv entry for food code {food_code}, portion code {portion_code}")
    return float(matches["Portion weight"].iloc[0]) * float(multiplier)


def code_match_tier(true_code, pred_code):
    """Bucket a prediction by how many leading digits of the 8-digit FNDDS code agree with
    ground truth -- FNDDS codes are hierarchical (leading digits = major food group, then
    progressively more specific subgroups), so a longer shared prefix means a more similar food
    even when it's not an exact match."""
    if pred_code is None:
        return "no_prediction"
    true_code, pred_code = str(true_code), str(pred_code)
    prefix_len = next((i for i in range(8) if true_code[i] != pred_code[i]), 8)
    if prefix_len == 8:
        return "exact"
    if prefix_len >= 2:
        return "close"
    if prefix_len == 1:
        return "far"
    return "mismatch"


def evaluate_row(row, llm, llm_vision, vectordb, food_weights_df, portions_xlsx, nutrients_xlsx):
    true_code = f"{int(row['FoodCode']):08d}"
    true_weight = true_portion_weight(
        row["FoodCode"], row["PortionCode"], row["PortionSubCode"], row["Multiplier"], food_weights_df
    )
    true_nutrients = compute_nutrients(true_code, true_weight, nutrients_xlsx)

    result = {"index": row.name, "true_code": true_code, "true_weight": true_weight}

    try:
        _, pred_code = infer_food_code(row["Link"], llm, llm_vision, vectordb)
    except RuntimeError as e:
        result.update({"pred_code": None, "pred_weight": None, "tier": "no_prediction", "note": str(e)})
        return result

    try:
        candidates = load_portion_candidates(pred_code, portions_xlsx)
        _, pred_weight = choose_portion(row["Link"], row["FC_Description"], candidates, MODELS["llm_vision"])
    except RuntimeError as e:
        result.update({"pred_code": pred_code, "pred_weight": None, "tier": code_match_tier(true_code, pred_code), "note": str(e)})
        return result

    pred_nutrients = compute_nutrients(pred_code, pred_weight, nutrients_xlsx)
    result.update(
        {
            "pred_code": pred_code,
            "pred_weight": pred_weight,
            "tier": code_match_tier(true_code, pred_code),
            "err_weight": abs(true_weight - pred_weight),
        }
    )
    for name in REPORT_NUTRIENTS:
        result[f"err_{name}"] = abs(true_nutrients.get(name, 0) - pred_nutrients.get(name, 0))
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark food-code/portion/nutrient accuracy against ASA24 ground truth.")
    sample_group = parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument("--sample_csv", help="Frozen ground-truth sample to evaluate (e.g. ../df_results_clean10.csv).")
    sample_group.add_argument("--source_csv", help="Full ground-truth table to draw a fresh random sample from (e.g. ../df_results.csv).")
    parser.add_argument("--n", type=int, default=10, help="Sample size when using --source_csv.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed when using --source_csv.")
    parser.add_argument("--food_weights_csv", default=DEFAULT_FOOD_WEIGHTS_CSV)
    parser.add_argument("--fndds_csv", default=DEFAULT_FNDDS_DESC_CSV)
    parser.add_argument("--portions_xlsx", default=DEFAULT_PORTIONS_XLSX)
    parser.add_argument("--nutrients_xlsx", default=DEFAULT_NUTRIENTS_XLSX)
    parser.add_argument("--persist_dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--output", required=True, help="CSV path to write per-row results to.")
    args = parser.parse_args()

    load_dotenv()
    llm = configure_chat_openai("llm", MODELS)
    llm_vision = configure_chat_openai("llm_vision", MODELS)
    embedding = OpenAIEmbeddings(
        model=MODELS["embedding"],
        base_url=LOCAL_VLLM_EMBED_BASE_URL,
        api_key="not-needed",
        check_embedding_ctx_length=False,
    )
    vectordb = build_vectordb(args.fndds_csv, args.persist_dir, embedding)
    food_weights_df = pd.read_csv(args.food_weights_csv)

    if args.sample_csv:
        sample = pd.read_csv(args.sample_csv)
    else:
        sample = pd.read_csv(args.source_csv).rename(columns={"FoodCodeCommon": "FoodCode"}).sample(
            n=args.n, random_state=args.seed
        )

    rows = []
    for _, row in sample.iterrows():
        print(f"[{row.name}] true={int(row['FoodCode']):08d} ({row['FC_Description']}) -- {row['Link']}", flush=True)
        try:
            result = evaluate_row(row, llm, llm_vision, vectordb, food_weights_df, args.portions_xlsx, args.nutrients_xlsx)
        except Exception as e:
            result = {"index": row.name, "true_code": f"{int(row['FoodCode']):08d}", "tier": "error", "note": str(e)}
        print(f"    -> pred={result.get('pred_code')} tier={result['tier']}", flush=True)
        rows.append(result)
        pd.DataFrame(rows).to_csv(args.output, index=False)

    df = pd.DataFrame(rows)
    print("\n=== Tier distribution ===")
    print(df["tier"].value_counts().to_string())
    print(f"\nExact-match accuracy: {(df['tier'] == 'exact').mean():.1%}")
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
