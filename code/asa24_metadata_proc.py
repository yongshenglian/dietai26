import argparse
from fractions import Fraction
from pathlib import Path

import pandas as pd


DEFAULT_METADATA_PATH = Path("../ASA24/ASA24_ImageMetadata.xlsx")
DEFAULT_IMAGE_DIR = Path("../ASA24/Asa24PortionImages")
DEFAULT_OUTPUT_DIR = Path("..")
DEFAULT_FOOD_WEIGHTS_PATH = Path("../FNDDS/FoodWeights.csv")


def format_food_code(value):
    if pd.isna(value):
        return pd.NA
    return f"{int(value):08d}"


def image_path(image_dir, file_name):
    return str((image_dir / file_name).resolve())


def split_portion(portion):
    amount, _, unit = str(portion).partition(" ")
    if "-" in amount:
        whole, fraction = amount.split("-", maxsplit=1)
        amount_value = Fraction(whole) + Fraction(fraction)
    else:
        amount_value = Fraction(amount)
    return float(amount_value), unit


def load_portion_references(metadata_path, image_dir):
    inventory = pd.read_excel(
        metadata_path,
        sheet_name="ImageFileInventory",
        usecols=["FileName", "Portion", "Display", "ImgDescription", "ImgTitle"],
    )
    codes = pd.read_excel(metadata_path, sheet_name="AssociatedCodes")
    codes = codes[codes["ImageType"].astype(str).str.upper() == "PORTION"].copy()

    references = codes.merge(
        inventory,
        left_on="Image",
        right_on="FileName",
        how="left",
        validate="many_to_one",
    )
    references = references.dropna(
        subset=[
            "Image",
            "FoodCode",
            "FC_Description",
            "Portion",
            "PortionCode",
            "PortionSubCode",
            "Multiplier",
        ]
    ).copy()

    references["FoodCode"] = references["FoodCode"].map(format_food_code)
    references["PortionCode"] = references["PortionCode"].astype("Int64")
    references["PortionSubCode"] = references["PortionSubCode"].astype("Int64")
    references["Link"] = references["Image"].map(lambda name: image_path(image_dir, name))
    references = references[references["Link"].map(lambda path: Path(path).is_file())]

    columns = [
        "FoodCode",
        "FC_Description",
        "Image",
        "Link",
        "Portion",
        "PortionCode",
        "PortionSubCode",
        "Multiplier",
        "Category",
        "Category Description",
        "ModCode",
        "Display",
        "ImgDescription",
        "ImgTitle",
        "ImageType",
    ]
    references = references[columns].drop_duplicates()
    return references.sort_values(
        by=["FoodCode", "Multiplier", "Portion", "Image"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def add_portion_choices(results, references):
    portion_choices = (
        references.sort_values(
            by=["FoodCode", "Multiplier", "Portion"],
            ascending=[True, False, True],
        )
        .groupby("FoodCode")["Portion"]
        .agg(lambda values: " ,".join(dict.fromkeys(values)))
    )
    label_parts = results["Portion"].map(split_portion)

    results = results.copy()
    results["PortionShot"] = results["FoodCodeCommon"].map(portion_choices)
    results["LabelAmount"] = label_parts.map(lambda parts: parts[0])
    results["LabelUnit"] = label_parts.map(lambda parts: parts[1])
    results["GPTFoodCode"] = pd.NA
    results["GPTFoodDescription"] = pd.NA
    return results


def filter_weight_compatible_references(references, food_weights_path):
    if food_weights_path is None or not food_weights_path.is_file():
        return references

    weights = pd.read_csv(
        food_weights_path,
        usecols=["Food code", "Portion code", "Portion weight"],
    ).dropna(subset=["Portion weight"])
    weights["FoodCode"] = weights["Food code"].map(format_food_code)
    weights["PortionCode"] = weights["Portion code"].astype("Int64")
    weights = weights[["FoodCode", "PortionCode"]].drop_duplicates()
    return references.merge(weights, on=["FoodCode", "PortionCode"], how="inner")


def build_test_results(references, sample_size, food_weights_path):
    references = filter_weight_compatible_references(references, food_weights_path)
    if references.empty:
        raise ValueError("No ASA24 rows matched the provided FNDDS FoodWeights table.")
    unique_food_rows = references.drop_duplicates(subset=["FoodCode"]).reset_index(drop=True)
    if sample_size >= len(unique_food_rows):
        test_rows = unique_food_rows
    else:
        step = (len(unique_food_rows) - 1) / max(sample_size - 1, 1)
        indices = [round(step * i) for i in range(sample_size)]
        test_rows = unique_food_rows.iloc[indices]
    results = test_rows[
        [
            "FoodCode",
            "FC_Description",
            "Image",
            "Link",
            "Portion",
            "PortionCode",
            "PortionSubCode",
            "Multiplier",
        ]
    ].copy()
    results = results.rename(columns={"FoodCode": "FoodCodeCommon"})
    return add_portion_choices(results, references)


def write_outputs(references, results, output_dir):
    references.to_csv(output_dir / "df_image_link.csv", index=False)
    results.to_csv(output_dir / "df_results.csv", index=False)
    results.to_csv(output_dir / "ASA24_GPTFoodCodes_portion.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ASA24 portion-reference CSV inputs used by DietAI24."
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--food-weights-path",
        type=Path,
        default=DEFAULT_FOOD_WEIGHTS_PATH,
        help="Optional FNDDS FoodWeights CSV used to seed only weight-compatible rows.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of ASA24 reference rows to seed into the test result CSVs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    references = load_portion_references(args.metadata_path, args.image_dir)
    results = build_test_results(references, args.sample_size, args.food_weights_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(references, results, args.output_dir)

    print(f"Wrote {len(references)} ASA24 portion-reference rows.")
    print(f"Wrote {len(results)} ASA24 test rows.")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
