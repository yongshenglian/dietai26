import argparse
import base64
import json
from pathlib import Path

import pandas as pd


DEFAULT_CSV_PATH = Path("../ASA24_GPTFoodCodes_nutrition.csv")
DEFAULT_REPORT_PATH = Path("../ASA24_GPTFoodCodes_nutrition_report.html")

CORE_COLUMNS = [
    "FoodCodeCommon",
    "FC_Description",
    "Portion",
    "GPTPortionDescription",
    "GPTPortionAmount",
    "TotalWeight",
    "Energy (kcal)",
    "Protein (g)",
    "Carbohydrate (g)",
    "Total Fat (g)",
    "Fiber, total dietary (g)",
    "Sodium (mg)",
]


def image_data_uri(value):
    if pd.isna(value):
        return None

    image_path = Path(str(value))
    if not image_path.is_file():
        return None

    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def load_report_data(csv_path):
    data = pd.read_csv(csv_path)
    data = data.loc[:, ~data.columns.str.startswith("Unnamed:")]
    data["_Thumbnail"] = data["Link"].map(image_data_uri) if "Link" in data else None
    data = data.where(pd.notna(data), None)
    columns = ["_Thumbnail"] + [column for column in data.columns if column != "_Thumbnail"]
    records = data[columns].to_dict(orient="records")
    core_columns = ["_Thumbnail"] + [column for column in CORE_COLUMNS if column in columns]
    return data, records, columns, core_columns


def summarize(data):
    def average(column):
        if column not in data:
            return None
        values = pd.to_numeric(data[column], errors="coerce")
        return None if values.dropna().empty else round(values.mean(), 2)

    return {
        "rows": len(data),
        "averageWeight": average("TotalWeight"),
        "averageEnergy": average("Energy (kcal)"),
        "averageProtein": average("Protein (g)"),
    }


def json_script(value):
    return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c")


def build_html(records, columns, core_columns, summary, csv_name):
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DietAI24 Nutrition Report</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #596575;
      --line: #d6dde6;
      --accent: #146c43;
      --accent-soft: #dff4e8;
      --warm: #9a4f00;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      background: var(--bg);
      color: var(--ink);
      margin: 0;
    }
    main {
      margin: 0 auto;
      max-width: 1480px;
      padding: 28px;
    }
    h1, h2, p { margin-top: 0; }
    h1 { font-size: 30px; margin-bottom: 8px; }
    .subhead { color: var(--muted); margin-bottom: 22px; }
    .metrics {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      margin-bottom: 18px;
    }
    .metric, .toolbar, .table-shell {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric { padding: 16px; }
    .metric span {
      color: var(--muted);
      display: block;
      font-size: 13px;
      margin-bottom: 7px;
    }
    .metric strong { font-size: 25px; font-weight: 650; }
    .toolbar {
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      justify-content: space-between;
      margin-bottom: 16px;
      padding: 14px;
    }
    .controls {
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }
    input[type="search"] {
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink);
      font: inherit;
      min-width: min(360px, 72vw);
      padding: 10px 12px;
    }
    label {
      align-items: center;
      color: var(--muted);
      display: inline-flex;
      gap: 7px;
      user-select: none;
    }
    .row-count { color: var(--muted); }
    .table-shell {
      max-height: calc(100vh - 300px);
      min-height: 430px;
      overflow: auto;
    }
    table {
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13px;
      min-width: 100%;
      width: max-content;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      max-width: 360px;
      padding: 10px 11px;
      text-align: left;
      vertical-align: top;
      white-space: normal;
    }
    th {
      background: #edf2f6;
      cursor: pointer;
      font-size: 12px;
      font-weight: 650;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    tbody tr:hover td { background: #f7fbf8; }
    td.number { font-variant-numeric: tabular-nums; text-align: right; }
    .thumb {
      background: #eef2f5;
      border: 1px solid var(--line);
      border-radius: 6px;
      display: block;
      height: 72px;
      object-fit: contain;
      width: 72px;
    }
    .empty-thumb {
      color: var(--muted);
      display: inline-block;
      min-width: 72px;
    }
    .portion {
      background: var(--accent-soft);
      border-radius: 4px;
      color: var(--accent);
      display: inline-block;
      font-weight: 600;
      padding: 3px 6px;
    }
    .warning { color: var(--warm); }
    @media (max-width: 720px) {
      main { padding: 16px; }
      h1 { font-size: 25px; }
      .table-shell { max-height: none; }
    }
  </style>
</head>
<body>
  <main>
    <h1>DietAI24 Nutrition Report</h1>
    <p class="subhead">Generated from <strong>__CSV_NAME__</strong>. Click a header to sort. Use the full-column toggle for all FNDDS nutrient fields.</p>
    <section class="metrics">
      <div class="metric"><span>Rows</span><strong id="rowMetric"></strong></div>
      <div class="metric"><span>Average weight</span><strong id="weightMetric"></strong></div>
      <div class="metric"><span>Average energy</span><strong id="energyMetric"></strong></div>
      <div class="metric"><span>Average protein</span><strong id="proteinMetric"></strong></div>
    </section>
    <section class="toolbar">
      <div class="controls">
        <input id="search" type="search" placeholder="Search food, code, portion, nutrient...">
        <label><input id="allColumns" type="checkbox"> Show all columns</label>
      </div>
      <div class="row-count" id="rowCount"></div>
    </section>
    <section class="table-shell">
      <table>
        <thead id="tableHead"></thead>
        <tbody id="tableBody"></tbody>
      </table>
    </section>
  </main>
  <script>
    const rows = __ROWS__;
    const allColumns = __COLUMNS__;
    const coreColumns = __CORE_COLUMNS__;
    const summary = __SUMMARY__;
    let visibleRows = [...rows];
    let sortColumn = "FC_Description";
    let sortDirection = 1;

    const numberFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });
    const prettyColumn = (column) => column === "_Thumbnail" ? "Image" : column.replaceAll("\n", " ");
    const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
    const currentColumns = () => document.querySelector("#allColumns").checked ? allColumns : coreColumns;
    const textValue = (value) => value === null || value === undefined ? "" : String(value);
    const metric = (value, unit) => value === null ? "n/a" : `${numberFormat.format(value)}${unit}`;

    function cell(column, value) {
      if (column === "_Thumbnail") {
        return value
          ? `<img class="thumb" src="${value}" alt="ASA24 portion image">`
          : `<span class="empty-thumb">No image</span>`;
      }
      if (value === null || value === undefined || value === "") return "";
      if (column === "GPTPortionDescription" || column === "Portion") {
        return `<span class="portion">${escapeHtml(textValue(value))}</span>`;
      }
      return escapeHtml(isNumber(value) ? numberFormat.format(value) : textValue(value));
    }

    function escapeHtml(value) {
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function compare(a, b) {
      const left = a[sortColumn];
      const right = b[sortColumn];
      if (left === right) return 0;
      if (left === null || left === undefined) return 1;
      if (right === null || right === undefined) return -1;
      if (isNumber(left) && isNumber(right)) return (left - right) * sortDirection;
      return String(left).localeCompare(String(right), undefined, { numeric: true }) * sortDirection;
    }

    function render() {
      const columns = currentColumns();
      visibleRows.sort(compare);
      document.querySelector("#tableHead").innerHTML = `<tr>${columns.map((column) =>
        `<th data-column="${escapeHtml(column)}">${escapeHtml(prettyColumn(column))}</th>`).join("")}</tr>`;
      document.querySelector("#tableBody").innerHTML = visibleRows.map((row) =>
        `<tr>${columns.map((column) => `<td class="${isNumber(row[column]) ? "number" : ""}">${cell(column, row[column])}</td>`).join("")}</tr>`
      ).join("");
      document.querySelector("#rowCount").textContent = `${visibleRows.length} of ${rows.length} rows`;
      document.querySelectorAll("th").forEach((header) => header.addEventListener("click", () => {
        const nextColumn = header.dataset.column;
        sortDirection = sortColumn === nextColumn ? sortDirection * -1 : 1;
        sortColumn = nextColumn;
        render();
      }));
    }

    function filterRows() {
      const query = document.querySelector("#search").value.trim().toLowerCase();
      visibleRows = query
        ? rows.filter((row) => Object.entries(row).some(([column, value]) =>
            column !== "_Thumbnail" && textValue(value).toLowerCase().includes(query)))
        : [...rows];
      render();
    }

    document.querySelector("#rowMetric").textContent = numberFormat.format(summary.rows);
    document.querySelector("#weightMetric").textContent = metric(summary.averageWeight, " g");
    document.querySelector("#energyMetric").textContent = metric(summary.averageEnergy, " kcal");
    document.querySelector("#proteinMetric").textContent = metric(summary.averageProtein, " g");
    document.querySelector("#search").addEventListener("input", filterRows);
    document.querySelector("#allColumns").addEventListener("change", render);
    render();
  </script>
</body>
</html>
"""
    return (
        template.replace("__CSV_NAME__", csv_name)
        .replace("__ROWS__", json_script(records))
        .replace("__COLUMNS__", json_script(columns))
        .replace("__CORE_COLUMNS__", json_script(core_columns))
        .replace("__SUMMARY__", json_script(summary))
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build a standalone DietAI24 nutrition HTML report.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    data, records, columns, core_columns = load_report_data(args.csv_path)
    html = build_html(records, columns, core_columns, summarize(data), args.csv_path.name)
    args.report_path.write_text(html, encoding="utf-8")
    print(f"Wrote {args.report_path.resolve()}")


if __name__ == "__main__":
    main()
