import argparse
import base64
import os
import re
from operator import itemgetter

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings

from config import MODELS, LOCAL_VLLM_BASE_URL, LOCAL_VLLM_EMBED_BASE_URL
from chagApp_openai import Vision
from rag_food_code import (
    configure_chat_openai,
    configure_retrievers,
    get_messages_from_url,
    load_data,
    setup_retrieval_prompt,
)
from rag_portion_size import create_prompt as create_portion_prompt

DEFAULT_FNDDS_DESC_CSV = "../FNDDS/2019-2020 FNDDS - Foods and Beverages.csv"
DEFAULT_PORTIONS_XLSX = "../FNDDS/2019-2020 FNDDS At A Glance - Portions and Weights.xlsx"
DEFAULT_NUTRIENTS_XLSX = "../FNDDS/2019-2020 FNDDS At A Glance - FNDDS Nutrient Values.xlsx"
DEFAULT_PERSIST_DIR = "../chroma_fndds_db"
COLLECTION_NAME = "fndds_food_descriptions"

REPORT_NUTRIENTS = [
    "Energy (kcal)",
    "Protein (g)",
    "Carbohydrate (g)",
    "Total Fat (g)",
    "Fiber, total dietary (g)",
    "Sodium (mg)",
]


def build_vectordb(fndds_csv, persist_dir, embedding):
    """Load a persisted Chroma DB of FNDDS food descriptions, building it once if missing."""
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding,
            persist_directory=persist_dir,
        )
    print(f"Embedding {fndds_csv} into {persist_dir} (one-time cost, ~5,600 rows)...")
    data = load_data(fndds_csv)
    return Chroma.from_documents(
        documents=data,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )


def plain_text_for_retrieval(text):
    """Strip markdown (bold, headers, bullets) so the query text reads like plain prose,
    matching FNDDS's description style closely enough for embedding similarity to work well."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^[#\-\*]+\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


CODE_PROMPT = ChatPromptTemplate.from_template(
    """A food image has already been described as: {description}

    Based only on that description and the following candidate FNDDS entries, identify the single
    best-matching eight-digit food code. Prefer a specific match (e.g. a named cut, dish, or
    preparation) over a generic "not further specified" catch-all category whenever a specific
    candidate reasonably fits -- catch-all categories should only be used when nothing more
    specific applies:

    {context}

    Reply with only the eight-digit code and no other text. If none of the candidates are a
    reasonable match, reply with exactly: No appropriate food codes found from the context information.
    """
)


def format_candidates(docs):
    """Render retrieved FNDDS Documents as plain 'Food code: X / Food description: Y' text,
    instead of dumping raw Document(metadata=..., page_content=...) reprs into the prompt.

    Generic "not further specified" catch-all entries are dropped whenever at least one specific
    candidate was retrieved. Measured: even with these sorted first and the prompt explicitly
    told to prefer a specific match, the model still picked the generic catch-all 3 of 5 times
    over genuinely present, on-topic specific candidates (e.g. "meat, not further specified"
    over "chicken thigh, roasted") -- prompt steering alone wasn't reliable enough, so this is
    enforced in code instead. Generic entries are kept only when nothing specific matched."""
    generic_markers = ("not further specified", "not specified subcategory")
    specific = [d for d in docs if not any(m in d.page_content.lower() for m in generic_markers)]
    generic = [d for d in docs if any(m in d.page_content.lower() for m in generic_markers)]
    kept = specific if specific else generic
    return "\n\n".join(doc.page_content for doc in kept)


def match_food_code(query_text, llm, vectordb):
    """RAG: retrieve candidate FNDDS entries for query_text and pick the single best match.
    Returns None (instead of raising) when no candidate is a reasonable match, since callers
    decomposing a plate into several ingredients need to skip a bad match, not abort the run."""
    retrieval_query = plain_text_for_retrieval(query_text)
    retriever = configure_retrievers(llm, vectordb, setup_retrieval_prompt())
    food_code_chain = (
        {
            "context": itemgetter("question") | retriever | format_candidates,
            "description": itemgetter("question"),
        }
        | CODE_PROMPT
        | llm
        | StrOutputParser()
    )
    response = food_code_chain.invoke({"question": retrieval_query})
    match = re.search(r"\d{8}", response)
    return match.group(0) if match else None


def infer_food_code(image_path, llm, llm_vision, vectordb):
    """Single-code mode: match the whole plate to one FNDDS code. Fast, but a multi-item plate
    (see analyze_ingredients) gets forced into whichever single code covers the most of it,
    silently dropping ingredients that code doesn't name."""
    description = llm_vision.invoke(get_messages_from_url(image_path)).content.strip()
    if description.startswith("I can't help to analyze this image."):
        raise RuntimeError(description)

    food_code = match_food_code(description, llm, vectordb)
    if food_code is None:
        raise RuntimeError("No appropriate food codes found from the context information.")
    return description, food_code


def _image_url(path_or_url):
    if os.path.isfile(path_or_url):
        with open(path_or_url, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    return path_or_url


def enumerate_ingredients(image_path, llm_vision):
    """Ask the vision model to list each distinct food item separately, instead of one
    holistic description of the whole plate -- this is what lets each component (e.g. a
    chicken thigh) get matched to its own FNDDS code and portion instead of being absorbed
    into (or dropped by) a single whole-plate match.

    Each line may start with a count (e.g. "2 chicken thighs") for discrete countable pieces
    of the same item -- FNDDS's own portion tables only have single-unit entries ("1 medium
    thigh"), no multi-piece options, so the count has to be captured here and applied as a
    multiplier later (see parse_quantity / analyze_ingredients)."""
    messages = [
        SystemMessage(
            content="You are an expert at analyzing food images with computer vision. You identify "
            "every distinct food item visible in a photo, one at a time."
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "List each distinct food item visible in this image, one per line. Name "
                    "the specific cut or variety when you can tell it apart (e.g. 'chicken thigh', "
                    "not just 'chicken' -- 'sweet potato', not just 'vegetable'). If multiple "
                    "discrete pieces of the exact same item are visible (e.g. two chicken thighs), "
                    "put them on one line prefixed with the count, e.g. '2 chicken thighs' -- don't "
                    "list them on separate lines and don't count non-discrete items like a scoop of "
                    "rice or a pile of vegetables. One item per line, no numbering, no bullets, no "
                    "other text. If you can't analyze the image, reply with exactly: I can't help to "
                    "analyze this image.",
                },
                {"type": "image_url", "image_url": {"url": _image_url(image_path)}},
            ]
        ),
    ]
    response = llm_vision.invoke(messages).content.strip()
    if response.startswith("I can't help to analyze this image."):
        raise RuntimeError(response)
    lines = (re.sub(r"^[\-\*\d.)]+\s*", "", line).strip() for line in response.split("\n"))
    return [line for line in lines if line]


def parse_quantity(item):
    """Split a leading count off an ingredient line ('2 chicken thighs' -> (2, 'chicken thighs')),
    since FNDDS's portion tables have no multi-piece entries -- the count has to be applied as a
    weight multiplier in code instead. No leading count means a single/non-discrete item (qty 1)."""
    match = re.match(r"^(\d+)\s+(.*)$", item)
    if match:
        return int(match.group(1)), match.group(2)
    return 1, item


def match_ingredients(image_path, llm, llm_vision, vectordb):
    """Enumerate the plate's ingredients and match each to an FNDDS food code. Shared by both
    portion-estimation strategies below -- neither needs its own copy of this."""
    raw_ingredients = enumerate_ingredients(image_path, llm_vision)
    print(f"Detected {len(raw_ingredients)} ingredient(s): {', '.join(raw_ingredients)}")

    matched = []
    for raw_name in raw_ingredients:
        quantity, name = parse_quantity(raw_name)
        print(f"  Matching '{name}'{f' (x{quantity})' if quantity != 1 else ''}...", flush=True)
        food_code = match_food_code(name, llm, vectordb)
        if food_code is None:
            print(f"    No FNDDS match found for '{name}' -- skipping.")
            continue
        matched.append({"raw_name": raw_name, "quantity": quantity, "name": name, "food_code": food_code})
    return matched


def analyze_ingredients(image_path, llm, llm_vision, vectordb, portions_xlsx, nutrients_xlsx):
    """Multi-item mode (label method): decompose the plate into separate ingredients, then for
    each one independently ask the vision model to pick its absolute size from that food code's
    own FNDDS portion labels ("1 medium thigh", etc.). Simple and works per-item, but empirically
    this absolute-size judgment is noisy (see choose_portion) -- analyze_ingredients_by_area is
    usually the better choice; this remains the default for backward compatibility and because it
    doesn't depend on an anchor ingredient being matched successfully."""
    for item in match_ingredients(image_path, llm, llm_vision, vectordb):
        try:
            candidates = load_portion_candidates(item["food_code"], portions_xlsx)
        except RuntimeError as e:
            print(f"    {e} -- skipping.")
            continue

        # Pass raw_name (with the count, e.g. "2 chicken thighs") here, not the cleaned name --
        # telling the vision model how many pieces are on the plate helps it judge each piece's
        # individual size (small/medium/large) rather than sizing the whole cluster as one unit.
        portion, unit_weight_g = choose_portion(image_path, item["raw_name"], candidates, MODELS["llm_vision"])
        weight_g = unit_weight_g * item["quantity"]
        nutrients = compute_nutrients(item["food_code"], weight_g, nutrients_xlsx)
        print(f"    Food code {item['food_code']}, portion '{portion}' x{item['quantity']} (~{weight_g:.0f} g total)")
        yield {
            "Ingredient": item["raw_name"],
            "FoodCode": item["food_code"],
            "Portion": portion,
            "Quantity": item["quantity"],
            "Weight_g": weight_g,
            **nutrients,
        }


def estimate_area_percentages(image_path, item_names, llm_vision):
    """Ask the vision model for every item's share of total food area in ONE pass. Measured to be
    far more consistent than asking each item's absolute size independently (see choose_portion):
    relative area comparison within one image doesn't need a real-world reference scale, which is
    exactly what absolute size estimation is missing. Returns {item_name: percent}."""
    items_str = ", ".join(item_names)
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Looking at this plate from directly above, estimate what percentage of "
                    f"the total food area (not counting empty plate) each of these items occupies, "
                    f"based purely on the area it covers in the image: {items_str}. List each item "
                    f"on its own line as 'item: X%', using the exact item names given. The "
                    f"percentages should sum to approximately 100%. No other text.",
                },
                {"type": "image_url", "image_url": {"url": _image_url(image_path)}},
            ]
        )
    ]
    response = llm_vision.invoke(messages).content.strip()
    percentages = {}
    for line in response.split("\n"):
        match = re.match(r"^(.+?):\s*(\d+(?:\.\d+)?)\s*%", line.strip())
        if match:
            percentages[match.group(1).strip()] = float(match.group(2))
    return percentages


def analyze_ingredients_by_area(image_path, llm, llm_vision, vectordb, portions_xlsx, nutrients_xlsx):
    """Multi-item mode (area method): estimate each ingredient's share of the plate's food area in
    one pass (reliable), anchor a total-plate-weight estimate on whichever matched ingredient has
    the largest area share (using the existing FNDDS-label portion picker just once), then allocate
    every ingredient's weight proportionally to its area share. Uses 1 portion-choice call instead
    of N, and empirically should be more accurate than N independent absolute-size guesses."""
    matched = match_ingredients(image_path, llm, llm_vision, vectordb)
    if not matched:
        return

    print("  Estimating area share of each ingredient...", flush=True)
    area_pct = estimate_area_percentages(image_path, [item["name"] for item in matched], llm_vision)

    anchor = max(matched, key=lambda item: area_pct.get(item["name"], 0))
    anchor_pct = area_pct.get(anchor["name"], 0)
    if anchor_pct <= 0:
        raise RuntimeError("Could not estimate an area percentage for any matched ingredient.")

    try:
        anchor_candidates = load_portion_candidates(anchor["food_code"], portions_xlsx)
    except RuntimeError as e:
        raise RuntimeError(f"Anchor ingredient '{anchor['name']}': {e}")
    print(f"  Anchoring on '{anchor['name']}' ({anchor_pct:.0f}% of area)...", flush=True)
    anchor_portion, anchor_unit_weight_g = choose_portion(
        image_path, anchor["raw_name"], anchor_candidates, MODELS["llm_vision"]
    )
    anchor_weight_g = anchor_unit_weight_g * anchor["quantity"]
    total_weight_g = anchor_weight_g / (anchor_pct / 100)
    print(f"    Anchor portion '{anchor_portion}' (~{anchor_weight_g:.0f} g) -> "
          f"total plate estimate ~{total_weight_g:.0f} g", flush=True)

    for item in matched:
        pct = area_pct.get(item["name"], 0)
        if pct <= 0:
            print(f"  No area estimate for '{item['name']}' -- skipping.")
            continue
        weight_g = total_weight_g * (pct / 100)
        nutrients = compute_nutrients(item["food_code"], weight_g, nutrients_xlsx)
        print(f"    {item['raw_name']}: {item['food_code']}, {pct:.0f}% of area (~{weight_g:.0f} g)")
        yield {
            "Ingredient": item["raw_name"],
            "FoodCode": item["food_code"],
            "AreaPercent": pct,
            "Weight_g": weight_g,
            **nutrients,
        }


def load_portion_candidates(food_code, portions_xlsx):
    df = pd.read_excel(portions_xlsx, sheet_name=0, header=1)
    matches = df[df["Food code"] == int(food_code)]
    if matches.empty:
        raise RuntimeError(f"No portion options found for food code {food_code}")
    return matches[["Portion description", "Portion weight (g)"]].drop_duplicates()


def choose_portion(image_path, description, candidates, vision_model_name):
    options = " ,".join(candidates["Portion description"].tolist())
    prompt = create_portion_prompt(description, options, type="shot")
    response = Vision(vision_model_name, base_url=LOCAL_VLLM_BASE_URL, api_key="not-needed").chat(prompt, image_path).strip()
    normalized = response.rstrip(".").strip()

    row = candidates[candidates["Portion description"] == normalized]
    if row.empty:
        row = candidates[
            candidates["Portion description"].str.contains(re.escape(normalized), case=False, na=False)
        ]
    if row.empty:
        raise RuntimeError(f"Could not match model portion choice '{response}' to a candidate")
    return response, float(row["Portion weight (g)"].iloc[0])


def compute_nutrients(food_code, weight_g, nutrients_xlsx):
    df = pd.read_excel(nutrients_xlsx, sheet_name="FNDDS Nutrient Values", skiprows=1)
    nutrient_cols = df.columns[4:]
    row = df[df["Food code"] == int(food_code)]
    if row.empty:
        raise RuntimeError(f"Food code {food_code} not found in nutrient table")
    per_100g = row.iloc[0][nutrient_cols]
    return (per_100g * weight_g * 0.01).to_dict()


def main():
    parser = argparse.ArgumentParser(description="Analyze a single food image: food code, portion, nutrients.")
    parser.add_argument("--image", required=True, help="Path or URL to the food image.")
    parser.add_argument(
        "--multi-ingredient",
        action="store_true",
        help="Decompose the plate into separate food items and match/weigh each one against its "
        "own FNDDS code, then sum. Slower, but doesn't lose ingredients a single whole-plate "
        "match wouldn't name (e.g. a chicken leg getting absorbed into a generic 'chicken and "
        "vegetables' code). Without this flag, the whole plate is matched to one FNDDS code.",
    )
    parser.add_argument(
        "--portion-method",
        choices=["label", "area"],
        default="area",
        help="Only used with --multi-ingredient. 'area' (default) estimates every ingredient's "
        "share of the plate's food area in one pass (measured to be far more consistent than "
        "per-item absolute-size guessing), anchors a total-plate-weight guess on the largest "
        "ingredient, and allocates the rest proportionally -- fewer model calls too (1 portion "
        "pick instead of N). 'label' asks the vision model to pick each ingredient's absolute "
        "size independently from its FNDDS portion labels -- simpler, but measured to be noisy "
        "since there's no size reference in the photo.",
    )
    parser.add_argument("--fndds_csv", default=DEFAULT_FNDDS_DESC_CSV)
    parser.add_argument("--portions_xlsx", default=DEFAULT_PORTIONS_XLSX)
    parser.add_argument("--nutrients_xlsx", default=DEFAULT_NUTRIENTS_XLSX)
    parser.add_argument("--persist_dir", default=DEFAULT_PERSIST_DIR, help="Where the FNDDS embeddings are cached.")
    parser.add_argument("--output", default=None, help="Optional CSV path to save the result row(s).")
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

    if args.multi_ingredient:
        print("Detecting ingredients (usually a few minutes for a multi-item plate)...", flush=True)
        analyze_fn = analyze_ingredients_by_area if args.portion_method == "area" else analyze_ingredients
        rows = list(analyze_fn(args.image, llm, llm_vision, vectordb, args.portions_xlsx, args.nutrients_xlsx))
        if not rows:
            raise RuntimeError("No ingredients could be matched to an FNDDS food code.")

        totals = {"Ingredient": "TOTAL", "FoodCode": ""}
        for key in ["Weight_g", *REPORT_NUTRIENTS]:
            totals[key] = sum(row.get(key, 0) for row in rows)

        print("\n=== Per-ingredient breakdown ===")
        for row in rows:
            detail = f"portion '{row['Portion']}'" if "Portion" in row else f"{row['AreaPercent']:.0f}% of area"
            print(f"  {row['Ingredient']}: {row['FoodCode']}, {detail} (~{row['Weight_g']:.0f} g)")

        print("\n=== Totals ===")
        print(f"  Total weight: {totals['Weight_g']:.0f} g")
        for name in REPORT_NUTRIENTS:
            print(f"  {name}: {totals[name]:.1f}")

        if args.output:
            pd.DataFrame([*rows, totals]).to_csv(args.output, index=False)
            print(f"Saved to {args.output}")
        return

    print("Describing image and identifying food code (usually 30-90s)...", flush=True)
    description, food_code = infer_food_code(args.image, llm, llm_vision, vectordb)
    print(f"Food description: {description}")
    print(f"FNDDS food code:  {food_code}")

    print("Selecting portion size...", flush=True)
    candidates = load_portion_candidates(food_code, args.portions_xlsx)
    portion, weight_g = choose_portion(args.image, description, candidates, MODELS["llm_vision"])
    print(f"Portion:          {portion} (~{weight_g:.0f} g)")

    nutrients = compute_nutrients(food_code, weight_g, args.nutrients_xlsx)
    print("Nutrients:")
    for name in REPORT_NUTRIENTS:
        if name in nutrients:
            print(f"  {name}: {nutrients[name]:.1f}")

    if args.output:
        row = {
            "Image": args.image,
            "FoodDescription": description,
            "FoodCode": food_code,
            "Portion": portion,
            "Weight_g": weight_g,
            **nutrients,
        }
        pd.DataFrame([row]).to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
