from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from PIL import Image, ImageDraw, ImageFont

from build_paper_package import add_markdown_to_docx, setup_doc


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PKG = ROOT / "manuscript"
CONTROL = PKG / "master_submission_control"
GIS_ROOT = Path("[public-source-root]/gis")
OUT_DIR = ROOT / "gis_energy_infrastructure"
FIG_DIR = ROOT / "figures"

OUT_JSON = CONTROL / "gis_energy_infrastructure_evidence.json"
OUT_MD = CONTROL / "gis_energy_infrastructure_evidence.md"
OUT_DOCX = CONTROL / "gis_energy_infrastructure_evidence.docx"
OUT_GRID_CSV = OUT_DIR / "china_grid_timeseries_summary.csv"
OUT_OSM_CSV = OUT_DIR / "osm_china_power_grid_summary.csv"
OUT_WRI_CSV = OUT_DIR / "wri_china_power_plant_fuel_summary.csv"
OUT_GEM_CSV = OUT_DIR / "gem_china_power_facility_summary.csv"
OUT_DERIVED_CSV = OUT_DIR / "gis_spatiotemporal_resource_heterogeneity_metrics.csv"

FIG_GRID = FIG_DIR / "paper3_fig14_china_grid_gis_externality.png"
FIG_RESOURCE = FIG_DIR / "dissertation_fig_gis_resource_mix_context.png"
FIG_DERIVED = FIG_DIR / "dissertation_fig_gis_spatiotemporal_resource_heterogeneity.png"

GRID_CSV_DIR = (
    GIS_ROOT
    / "China_Power_Grid_Timeseries_2015_2025"
    / "01_china_power_transmission_network_2015_2020_2025"
    / "03_delivery_organized_shp_csv"
    / "03_csv_by_year"
)
OSM_CSV_DIR = GIS_ROOT / "OSM_China_Power_Grid" / "02_delivery_shp_csv" / "02_mainland_csv"
WRI_GEOJSON = GIS_ROOT / "WRI_China_Power_Plants" / "01_wri_global_power_plant_database" / "china_power_plants_wri.geojson"
GEM_INTEGRATED = (
    GIS_ROOT
    / "GEM_China_Energy_Projects"
    / "02_china_filtered_tables"
    / "integrated-power-tracker"
    / "Global-Integrated-Power-March-2026-II__Power facilities__china.csv"
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def safe_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "null", "--"}:
        return None
    try:
        x = float(text)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def parse_voltage(value) -> list[float]:
    text = str(value or "")
    vals = []
    for token in re.split(r"[;,/|\s]+", text):
        x = safe_float(token)
        if x is not None and x > 0:
            vals.append(x)
    return vals


def count_csv_rows(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return 0, []
        return sum(1 for _ in reader), header


def profile_grid_timeseries() -> list[dict]:
    rows: list[dict] = []
    for year_dir in sorted(p for p in GRID_CSV_DIR.iterdir() if p.is_dir()):
        year = int(year_dir.name)
        for path in sorted(year_dir.glob("*.csv")):
            feature = path.stem.rsplit("_", 1)[0]
            nrows, header = count_csv_rows(path)
            voltage_col = "voltage" if "voltage" in header else None
            status_col = "status" if "status" in header else None
            voltages: list[float] = []
            status_counts: Counter[str] = Counter()
            if voltage_col or status_col:
                for chunk in pd.read_csv(path, usecols=[c for c in [voltage_col, status_col] if c], chunksize=100_000, low_memory=False):
                    if voltage_col:
                        for item in chunk[voltage_col].dropna().astype(str):
                            vals = parse_voltage(item)
                            if vals:
                                voltages.append(max(vals))
                    if status_col:
                        status_counts.update(compact(v) or "missing" for v in chunk[status_col].fillna("missing"))
            rows.append(
                {
                    "year": year,
                    "feature": feature,
                    "rows": nrows,
                    "columns": len(header),
                    "voltage_non_null": len(voltages),
                    "max_voltage_kv": round(max(voltages) / 1000, 1) if voltages else "",
                    "median_voltage_kv": round(float(pd.Series(voltages).median()) / 1000, 1) if voltages else "",
                    "share_500kv_plus": round(sum(v >= 500_000 for v in voltages) / len(voltages), 4) if voltages else "",
                    "top_status": "; ".join(f"{k}:{v}" for k, v in status_counts.most_common(4)),
                    "source_file": path.as_posix(),
                }
            )
    return rows


def parse_osm_generator_source(other_tags: str) -> str:
    text = str(other_tags or "")
    match = re.search(r"generator:source=([^;]+)", text)
    return match.group(1).strip() if match else "unknown"


def profile_osm() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(OSM_CSV_DIR.glob("power_*.csv")):
        nrows, header = count_csv_rows(path)
        record = {
            "dataset": path.stem,
            "rows": nrows,
            "columns": len(header),
            "source_file": path.as_posix(),
            "key_distribution": "",
            "voltage_non_null_share": "",
        }
        if path.name == "power_points_generator.csv":
            source_counts: Counter[str] = Counter()
            for chunk in pd.read_csv(path, usecols=["other_tags"], chunksize=100_000, low_memory=False):
                source_counts.update(parse_osm_generator_source(v) for v in chunk["other_tags"].fillna(""))
            record["key_distribution"] = "; ".join(f"{k}:{v}" for k, v in source_counts.most_common(8))
        elif "voltage" in header:
            non_null = 0
            for chunk in pd.read_csv(path, usecols=["voltage"], chunksize=100_000, low_memory=False):
                non_null += int(chunk["voltage"].fillna("").astype(str).str.strip().ne("").sum())
            record["voltage_non_null_share"] = round(non_null / nrows, 4) if nrows else ""
        rows.append(record)
    return rows


def profile_wri() -> list[dict]:
    data = json.loads(WRI_GEOJSON.read_text(encoding="utf-8"))
    totals: dict[str, dict] = defaultdict(lambda: {"plant_count": 0, "capacity_mw": 0.0})
    missing_capacity = 0
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        fuel = compact(props.get("primary_fuel") or "Unknown")
        cap = safe_float(props.get("capacity_mw"))
        totals[fuel]["plant_count"] += 1
        if cap is None:
            missing_capacity += 1
        else:
            totals[fuel]["capacity_mw"] += cap
    rows = []
    for fuel, vals in sorted(totals.items(), key=lambda item: item[1]["capacity_mw"], reverse=True):
        rows.append(
            {
                "fuel": fuel,
                "plant_count": vals["plant_count"],
                "capacity_mw": round(vals["capacity_mw"], 3),
                "capacity_gw": round(vals["capacity_mw"] / 1000, 3),
                "source_file": WRI_GEOJSON.as_posix(),
            }
        )
    rows.append(
        {
            "fuel": "__metadata__",
            "plant_count": len(data.get("features", [])),
            "capacity_mw": "",
            "capacity_gw": "",
            "source_file": f"missing_capacity_records={missing_capacity}",
        }
    )
    return rows


def profile_gem_integrated() -> list[dict]:
    usecols = ["Type", "Capacity (MW)", "Status", "Start year", "Latitude", "Longitude", "Location accuracy"]
    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {"rows": 0, "capacity_mw": 0.0, "missing_capacity": 0, "exact_or_approx": 0})
    for chunk in pd.read_csv(GEM_INTEGRATED, usecols=usecols, chunksize=150_000, low_memory=False):
        for _, row in chunk.iterrows():
            typ = compact(row.get("Type") or "unknown")
            status = compact(row.get("Status") or "missing")
            key = (typ, status)
            agg[key]["rows"] += 1
            cap = safe_float(row.get("Capacity (MW)"))
            if cap is None:
                agg[key]["missing_capacity"] += 1
            else:
                agg[key]["capacity_mw"] += cap
            acc = compact(row.get("Location accuracy")).lower()
            if acc in {"exact", "approximate"}:
                agg[key]["exact_or_approx"] += 1
    rows = []
    for (typ, status), vals in sorted(agg.items(), key=lambda item: item[1]["capacity_mw"], reverse=True):
        rows.append(
            {
                "type": typ,
                "status": status,
                "rows": vals["rows"],
                "capacity_mw": round(vals["capacity_mw"], 3),
                "capacity_gw": round(vals["capacity_mw"] / 1000, 3),
                "missing_capacity": vals["missing_capacity"],
                "exact_or_approx_location_share": round(vals["exact_or_approx"] / vals["rows"], 4) if vals["rows"] else "",
                "source_file": GEM_INTEGRATED.as_posix(),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, size=24, fill="#0f172a", bold=False, anchor=None):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor=anchor)


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], y_max: float, y_label: str):
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill="#94a3b8", width=2)
    draw.line((x0, y0, x0, y1), fill="#94a3b8", width=2)
    for i in range(5):
        y = y1 - (y1 - y0) * i / 4
        val = y_max * i / 4
        draw.line((x0, y, x1, y), fill="#e2e8f0", width=1)
        draw_text(draw, (x0 - 10, y), f"{val:,.0f}", size=15, fill="#64748b", anchor="rm")
    draw_text(draw, (x0, y0 - 18), y_label, size=16, fill="#475569")


def plot_grid(grid_rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    feature_labels = {
        "01_power_transmission_lines": "Transmission lines",
        "02_substations": "Substations",
        "04_grid_links": "Grid links",
    }
    colors = {"Transmission lines": "#2b6cb0", "Substations": "#0f766e", "Grid links": "#b45309"}
    image = Image.new("RGB", (1800, 860), "white")
    draw = ImageDraw.Draw(image)
    draw_text(draw, (90, 55), "Open GIS evidence for network-scale VPP decision context", size=36, bold=True)
    draw_text(draw, (90, 95), "China transmission-network snapshots and high-voltage evidence", size=20, fill="#64748b")

    left = (120, 190, 910, 690)
    series = {}
    for feature, label in feature_labels.items():
        xs, ys = [], []
        for row in sorted([r for r in grid_rows if r["feature"] == feature], key=lambda r: r["year"]):
            xs.append(row["year"])
            ys.append(row["rows"])
        if xs and ys:
            series[label] = (xs, ys)
    y_max = max(max(ys) for _, ys in series.values()) * 1.1
    draw_text(draw, (left[0], left[1] - 42), "China grid GIS record growth", size=24, bold=True)
    draw_axes(draw, left, y_max, "Feature records")
    years = [2015, 2020, 2025]
    x0, y0, x1, y1 = left
    x_pos = {year: x0 + (x1 - x0) * idx / (len(years) - 1) for idx, year in enumerate(years)}
    for year in years:
        draw_text(draw, (x_pos[year], y1 + 28), str(year), size=17, fill="#475569", anchor="mm")
    for label, (xs, ys) in series.items():
        pts = []
        for x, y in zip(xs, ys):
            px = x_pos[int(x)]
            py = y1 - (y / y_max) * (y1 - y0)
            pts.append((px, py))
        if len(pts) > 1:
            draw.line(pts, fill=colors[label], width=5)
        for px, py in pts:
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=colors[label], outline="white", width=2)
    for i, label in enumerate(series):
        lx = x0 + i * 230
        draw.rectangle((lx, y1 + 62, lx + 24, y1 + 78), fill=colors[label])
        draw_text(draw, (lx + 34, y1 + 70), label, size=17, fill="#334155", anchor="lm")

    latest = [r for r in grid_rows if r["year"] == 2025 and r["feature"] in feature_labels]
    labels = [feature_labels[r["feature"]] for r in latest]
    values = [float(r["share_500kv_plus"] or 0) * 100 for r in latest]
    right = (1060, 190, 1680, 690)
    draw_text(draw, (right[0], right[1] - 42), "2025 high-voltage evidence", size=24, bold=True)
    draw_axes(draw, right, max(values + [1]) * 1.25, "Share >= 500 kV (%)")
    bar_w = 110
    gap = 90
    rx0, ry0, rx1, ry1 = right
    for i, (label, value) in enumerate(zip(labels, values)):
        x = rx0 + 100 + i * (bar_w + gap)
        h = (value / (max(values + [1]) * 1.25)) * (ry1 - ry0)
        draw.rounded_rectangle((x, ry1 - h, x + bar_w, ry1), radius=8, fill=colors[label])
        draw_text(draw, (x + bar_w / 2, ry1 - h - 18), f"{value:.1f}%", size=18, bold=True, fill="#0f172a", anchor="mm")
        draw_text(draw, (x + bar_w / 2, ry1 + 32), label.replace(" ", "\n"), size=15, fill="#475569", anchor="ma")

    draw_text(
        draw,
        (90, 812),
        "Source: local copies of China power transmission network GIS snapshots; application-context evidence only until source/license gates are closed.",
        size=16,
        fill="#64748b",
    )
    image.save(FIG_GRID)


def plot_resource_mix(wri_rows: list[dict], gem_rows: list[dict]) -> None:
    wri = [r for r in wri_rows if not r["fuel"].startswith("__")][:8]
    gem_by_type: dict[str, float] = defaultdict(float)
    for row in gem_rows:
        if row["status"].lower() == "operating":
            gem_by_type[row["type"]] += float(row["capacity_mw"] or 0)
    gem_top = sorted(gem_by_type.items(), key=lambda item: item[1], reverse=True)[:8]
    image = Image.new("RGB", (1800, 900), "white")
    draw = ImageDraw.Draw(image)
    draw_text(draw, (90, 55), "China resource-mix GIS metadata for thesis context", size=36, bold=True)
    draw_text(draw, (90, 95), "Open plant and project metadata summarized as scenario-design evidence", size=20, fill="#64748b")

    def draw_barh(items, box, title, color):
        x0, y0, x1, y1 = box
        draw_text(draw, (x0, y0 - 42), title, size=24, bold=True)
        max_val = max([v for _, v in items] + [1])
        row_h = (y1 - y0) / len(items)
        for idx, (label, value) in enumerate(items):
            y = y0 + idx * row_h + 8
            bar_len = (value / max_val) * (x1 - x0 - 260)
            draw_text(draw, (x0, y + row_h * 0.35), label[:24], size=17, fill="#334155", anchor="lm")
            draw.rounded_rectangle((x0 + 215, y, x0 + 215 + bar_len, y + row_h * 0.55), radius=8, fill=color)
            draw_text(draw, (x0 + 225 + bar_len, y + row_h * 0.28), f"{value:,.1f} GW", size=16, fill="#0f172a", anchor="lm")
        draw.line((x0 + 215, y1 + 8, x1, y1 + 8), fill="#cbd5e1", width=2)

    draw_barh([(r["fuel"], float(r["capacity_gw"] or 0)) for r in wri], (90, 190, 850, 720), "WRI plant capacity by fuel", "#334155")
    draw_barh([(k, v / 1000) for k, v in gem_top], (980, 190, 1710, 720), "GEM operating facilities by type", "#0f766e")
    draw_text(
        draw,
        (90, 838),
        "Source: WRI China power plants GeoJSON and GEM China integrated power tracker local tables; background and scenario-design evidence, not current model-training data.",
        size=16,
        fill="#64748b",
    )
    image.save(FIG_RESOURCE)


def pct_change(start: float, end: float) -> float:
    if start == 0:
        return 0.0
    return (end - start) / start * 100


def shannon_entropy(shares: list[float]) -> float:
    vals = [v for v in shares if v > 0]
    return -sum(v * math.log(v) for v in vals)


def parse_distribution(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(text or "").split(";"):
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        num = safe_float(val)
        if num is not None:
            out[compact(key)] = num
    return out


def build_derived_metrics(grid_rows: list[dict], osm_rows: list[dict], wri_rows: list[dict], gem_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    by_feature_year = {(r["feature"], int(r["year"])): r for r in grid_rows}
    feature_labels = {
        "01_power_transmission_lines": "transmission_line_records",
        "02_substations": "substation_records",
        "04_grid_links": "grid_link_records",
    }
    for feature, label in feature_labels.items():
        r2015 = by_feature_year.get((feature, 2015), {})
        r2025 = by_feature_year.get((feature, 2025), {})
        start = float(r2015.get("rows") or 0)
        end = float(r2025.get("rows") or 0)
        rows.append(
            {
                "metric_group": "grid_temporal_drift",
                "metric": f"{label}_2015_to_2025_growth_pct",
                "value": round(pct_change(start, end), 3),
                "unit": "percent",
                "interpretation": "Infrastructure-context drift across the available China grid snapshots; motivates rolling validation and time-aware graph construction.",
            }
        )
        if r2025.get("share_500kv_plus") != "":
            rows.append(
                {
                    "metric_group": "grid_temporal_drift",
                    "metric": f"{label}_2025_share_500kv_plus",
                    "value": round(float(r2025["share_500kv_plus"]) * 100, 3),
                    "unit": "percent",
                    "interpretation": "High-voltage coverage proxy; supports the claim that VPP decisions may face network-scale coupling rather than isolated device control.",
                }
            )

    generator = next((r for r in osm_rows if r["dataset"] == "power_points_generator"), None)
    if generator:
        dist = parse_distribution(generator.get("key_distribution", ""))
        total = float(generator.get("rows") or 0)
        renewable = sum(dist.get(k, 0.0) for k in ["wind", "solar", "hydro", "battery"])
        shares = [v / total for v in dist.values()] if total else []
        rows.extend(
            [
                {
                    "metric_group": "osm_generator_mix",
                    "metric": "osm_generator_points_count",
                    "value": int(total),
                    "unit": "records",
                    "interpretation": "Generator-point context extracted from OSM-derived mainland power-grid records; used only as coverage/context evidence.",
                },
                {
                    "metric_group": "osm_generator_mix",
                    "metric": "osm_generator_renewable_or_storage_share",
                    "value": round(renewable / total * 100, 3) if total else 0,
                    "unit": "percent",
                    "interpretation": "Resource-mix heterogeneity proxy; motivates load-transfer and VPP resource-state interfaces.",
                },
                {
                    "metric_group": "osm_generator_mix",
                    "metric": "osm_generator_source_entropy",
                    "value": round(shannon_entropy(shares), 4),
                    "unit": "nats",
                    "interpretation": "Diversity of generator source tags; not a reconciled capacity statistic and not used for model training.",
                },
            ]
        )

    wri = [r for r in wri_rows if not str(r["fuel"]).startswith("__")]
    wri_total = sum(float(r["capacity_gw"] or 0) for r in wri)
    wri_renewable = sum(float(r["capacity_gw"] or 0) for r in wri if str(r["fuel"]).lower() in {"hydro", "solar", "wind", "geothermal"})
    wri_shares = [float(r["capacity_gw"] or 0) / wri_total for r in wri] if wri_total else []
    rows.extend(
        [
            {
                "metric_group": "wri_capacity_mix",
                "metric": "wri_capacity_total_gw",
                "value": round(wri_total, 3),
                "unit": "GW",
                "interpretation": "Open plant-capacity context from WRI; provides resource-scale background only.",
            },
            {
                "metric_group": "wri_capacity_mix",
                "metric": "wri_renewable_capacity_share",
                "value": round(wri_renewable / wri_total * 100, 3) if wri_total else 0,
                "unit": "percent",
                "interpretation": "Capacity-mix heterogeneity proxy across fuel classes.",
            },
            {
                "metric_group": "wri_capacity_mix",
                "metric": "wri_capacity_mix_entropy",
                "value": round(shannon_entropy(wri_shares), 4),
                "unit": "nats",
                "interpretation": "Capacity-weighted resource diversity; supports scenario-design discussion without becoming a training label.",
            },
        ]
    )

    operating = [r for r in gem_rows if str(r["status"]).lower() == "operating"]
    gem_total = sum(float(r["capacity_gw"] or 0) for r in operating)
    flexible_or_variable = sum(
        float(r["capacity_gw"] or 0)
        for r in operating
        if str(r["type"]).lower() in {"utility-scale solar", "wind", "hydropower", "oil/gas", "battery storage"}
    )
    gem_shares = [float(r["capacity_gw"] or 0) / gem_total for r in operating if gem_total]
    rows.extend(
        [
            {
                "metric_group": "gem_operating_mix",
                "metric": "gem_operating_capacity_gw",
                "value": round(gem_total, 3),
                "unit": "GW",
                "interpretation": "Operating-capacity context from the local China GEM integrated-power table.",
            },
            {
                "metric_group": "gem_operating_mix",
                "metric": "gem_variable_or_dispatchable_context_share",
                "value": round(flexible_or_variable / gem_total * 100, 3) if gem_total else 0,
                "unit": "percent",
                "interpretation": "Broad resource-context share used to motivate VPP state heterogeneity; not a dispatchable-capacity estimate.",
            },
            {
                "metric_group": "gem_operating_mix",
                "metric": "gem_operating_type_entropy",
                "value": round(shannon_entropy(gem_shares), 4),
                "unit": "nats",
                "interpretation": "Operating resource-type diversity proxy for scenario realism and source-domain heterogeneity.",
            },
        ]
    )
    return rows


def plot_derived_metrics(rows: list[dict]) -> None:
    selected = [
        ("Line growth", "transmission_line_records_2015_to_2025_growth_pct", "%", "#2b6cb0"),
        ("Grid-link growth", "grid_link_records_2015_to_2025_growth_pct", "%", "#0f766e"),
        ("OSM renewable/storage tags", "osm_generator_renewable_or_storage_share", "%", "#b45309"),
        ("WRI renewable capacity", "wri_renewable_capacity_share", "%", "#4c1d95"),
        ("GEM variable/dispatchable context", "gem_variable_or_dispatchable_context_share", "%", "#be123c"),
    ]
    value_by_metric = {r["metric"]: float(r["value"]) for r in rows}
    image = Image.new("RGB", (1800, 940), "white")
    draw = ImageDraw.Draw(image)
    draw_text(draw, (90, 58), "GIS-derived spatiotemporal and resource heterogeneity metrics", size=34, bold=True)
    draw_text(draw, (90, 98), "Aggregate context variables used for problem framing, not model training", size=20, fill="#64748b")
    x0, y0, x1, y1 = 170, 210, 1660, 735
    max_val = max([value_by_metric.get(metric, 0.0) for _, metric, _, _ in selected] + [1]) * 1.12
    draw_axes(draw, (x0, y0, x1, y1), max_val, "Percent or percentage change")
    bar_w = 150
    gap = 125
    for idx, (label, metric, unit, color) in enumerate(selected):
        value = value_by_metric.get(metric, 0.0)
        x = x0 + 105 + idx * (bar_w + gap)
        h = value / max_val * (y1 - y0)
        draw.rounded_rectangle((x, y1 - h, x + bar_w, y1), radius=8, fill=color)
        draw_text(draw, (x + bar_w / 2, y1 - h - 20), f"{value:.1f}{unit}", size=19, bold=True, anchor="mm")
        words = label.split()
        wrapped = "\n".join([" ".join(words[:2]), " ".join(words[2:])]) if len(words) > 2 else label
        draw_text(draw, (x + bar_w / 2, y1 + 40), wrapped, size=17, fill="#334155", anchor="ma")
    entropy_rows = [r for r in rows if r["metric"].endswith("_entropy")]
    draw_text(draw, (90, 832), "Diversity proxies: " + "; ".join(f"{r['metric']}={float(r['value']):.3f} {r['unit']}" for r in entropy_rows), size=17, fill="#475569")
    draw_text(draw, (90, 866), "Boundary: aggregate GIS metrics support graph, transfer, and VPP scenario motivation; public OPSD/UCI experiments remain the reproducible causal evidence.", size=16, fill="#64748b")
    image.save(FIG_DERIVED)


def md_table(rows: list[dict], columns: list[str], limit: int | None = None) -> str:
    subset = rows if limit is None else rows[:limit]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in subset:
        lines.append("|" + "|".join(compact(row.get(c, ""))[:180] for c in columns) + "|")
    return "\n".join(lines)


def build_payload() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_rows = profile_grid_timeseries()
    osm_rows = profile_osm()
    wri_rows = profile_wri()
    gem_rows = profile_gem_integrated()
    derived_rows = build_derived_metrics(grid_rows, osm_rows, wri_rows, gem_rows)
    write_csv(OUT_GRID_CSV, grid_rows)
    write_csv(OUT_OSM_CSV, osm_rows)
    write_csv(OUT_WRI_CSV, wri_rows)
    write_csv(OUT_GEM_CSV, gem_rows)
    write_csv(OUT_DERIVED_CSV, derived_rows)
    plot_grid(grid_rows)
    plot_resource_mix(wri_rows, gem_rows)
    plot_derived_metrics(derived_rows)

    grid_latest = {r["feature"]: r for r in grid_rows if r["year"] == 2025}
    osm_total = sum(int(r["rows"]) for r in osm_rows)
    wri_meta = next(r for r in wri_rows if r["fuel"] == "__metadata__")
    gem_operating_capacity = sum(float(r["capacity_mw"] or 0) for r in gem_rows if r["status"].lower() == "operating")

    return {
        "generated": date.today().isoformat(),
        "status": "PASS_WITH_LICENSE_AND_SOURCE_PAGE_GATES",
        "purpose": "GIS energy-infrastructure evidence layer for dissertation context and bounded journal external-validity discussion.",
        "source_root": GIS_ROOT.as_posix(),
        "summary": {
            "grid_2025_transmission_line_records": grid_latest.get("01_power_transmission_lines", {}).get("rows", 0),
            "grid_2025_substation_records": grid_latest.get("02_substations", {}).get("rows", 0),
            "grid_2025_grid_link_records": grid_latest.get("04_grid_links", {}).get("rows", 0),
            "osm_mainland_power_records": osm_total,
            "wri_china_power_plant_records": wri_meta["plant_count"],
            "gem_operating_capacity_gw": round(gem_operating_capacity / 1000, 3),
            "grid_line_record_growth_2015_2025_pct": next(
                r["value"] for r in derived_rows if r["metric"] == "transmission_line_records_2015_to_2025_growth_pct"
            ),
            "osm_generator_renewable_or_storage_share_pct": next(
                r["value"] for r in derived_rows if r["metric"] == "osm_generator_renewable_or_storage_share"
            ),
            "wri_renewable_capacity_share_pct": next(
                r["value"] for r in derived_rows if r["metric"] == "wri_renewable_capacity_share"
            ),
        },
        "outputs": {
            "grid_summary_csv": rel(OUT_GRID_CSV),
            "osm_summary_csv": rel(OUT_OSM_CSV),
            "wri_summary_csv": rel(OUT_WRI_CSV),
            "gem_summary_csv": rel(OUT_GEM_CSV),
            "derived_metrics_csv": rel(OUT_DERIVED_CSV),
            "grid_context_figure": rel(FIG_GRID),
            "resource_mix_figure": rel(FIG_RESOURCE),
            "derived_metrics_figure": rel(FIG_DERIVED),
        },
        "grid_rows": grid_rows,
        "osm_rows": osm_rows,
        "wri_rows": wri_rows,
        "gem_rows": gem_rows,
        "derived_rows": derived_rows,
        "paper_use": [
            {
                "paper": "Paper 1 price forecasting",
                "use": "Graph-prior and regional heterogeneity motivation only; do not turn GIS metadata into a price target.",
            },
            {
                "paper": "Paper 2 load forecasting",
                "use": "Resource and network heterogeneity motivation for cross-domain load transfer; keep quantitative claims on UCI/OPSD/Shandong validated load data.",
            },
            {
                "paper": "Paper 3 VPP decision-focused bidding",
                "use": "External-validity evidence that VPP decision policies operate in a large resource-network context with renewables, substations, transmission lines, and storage-adjacent assets.",
            },
            {
                "paper": "Doctoral dissertation",
                "use": "Chapter 2 infrastructure background, Chapter 5 VPP scenario design, and Chapter 6 data-governance appendix.",
            },
        ],
        "hard_boundaries": [
            "The GIS layer is not used to train the public OPSD/UCI journal baselines in the current package.",
            "OSM-derived records require ODbL attribution/share-alike review before redistribution.",
            "GEM materials may carry non-commercial or tracker-specific use caveats; keep raw GEM files out of public supplements until confirmed.",
            "Figshare/WRI-style open-license sources can be cited with attribution, but exact source pages and versions still need final manuscript reference checks.",
        ],
    }


def build_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# GIS Energy-Infrastructure Evidence Audit",
        "",
        f"Generated: {payload['generated']}",
        f"Status: {payload['status']}",
        "",
        payload["purpose"],
        "",
        "## Summary",
        "",
        f"- 2025 transmission-line records: {summary['grid_2025_transmission_line_records']}",
        f"- 2025 substation records: {summary['grid_2025_substation_records']}",
        f"- 2025 grid-link records: {summary['grid_2025_grid_link_records']}",
        f"- OSM mainland power records profiled: {summary['osm_mainland_power_records']}",
        f"- WRI China power plant records: {summary['wri_china_power_plant_records']}",
        f"- GEM operating facility capacity represented: {summary['gem_operating_capacity_gw']} GW",
        f"- 2015-2025 transmission-line record growth: {summary['grid_line_record_growth_2015_2025_pct']}%",
        f"- OSM generator renewable/storage source-tag share: {summary['osm_generator_renewable_or_storage_share_pct']}%",
        f"- WRI renewable capacity share: {summary['wri_renewable_capacity_share_pct']}%",
        "",
        "## China Grid Time-Series Snapshot",
        "",
        md_table(
            payload["grid_rows"],
            ["year", "feature", "rows", "columns", "voltage_non_null", "max_voltage_kv", "median_voltage_kv", "share_500kv_plus", "top_status"],
        ),
        "",
        f"![Figure. Open GIS evidence for network-scale VPP decision context.]({FIG_GRID})",
        "",
        "## OSM Mainland Power Extraction",
        "",
        md_table(payload["osm_rows"], ["dataset", "rows", "columns", "key_distribution", "voltage_non_null_share"]),
        "",
        "## WRI China Power Plants",
        "",
        md_table(payload["wri_rows"], ["fuel", "plant_count", "capacity_gw"], limit=12),
        "",
        "## GEM Integrated Power Tracker",
        "",
        md_table(payload["gem_rows"], ["type", "status", "rows", "capacity_gw", "missing_capacity", "exact_or_approx_location_share"], limit=18),
        "",
        f"![Figure. China resource-mix GIS metadata for thesis context.]({FIG_RESOURCE})",
        "",
        "## Derived Spatiotemporal and Resource-Heterogeneity Metrics",
        "",
        md_table(payload["derived_rows"], ["metric_group", "metric", "value", "unit", "interpretation"]),
        "",
        f"![Figure. GIS-derived spatiotemporal and resource heterogeneity metrics.]({FIG_DERIVED})",
        "",
        "## Paper Integration Plan",
        "",
        md_table(payload["paper_use"], ["paper", "use"]),
        "",
        "## Hard Boundaries",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["hard_boundaries"])
    lines.extend(
        [
            "",
            "## Recommended Manuscript Wording",
            "",
            "The GIS layer should be described as infrastructure-context evidence: it documents large-scale network, generation, and resource heterogeneity that motivates graph-aware forecasting and decision-focused VPP evaluation. It should not be described as a new model-training dataset for the current public OPSD/UCI experiments.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    CONTROL.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = build_markdown(payload)
    OUT_MD.write_text(md, encoding="utf-8")
    doc = Document()
    setup_doc(doc, "GIS Energy-Infrastructure Evidence Audit")
    add_markdown_to_docx(doc, md)
    doc.save(OUT_DOCX)
    print(payload["status"], payload["summary"])
    print(OUT_MD)


if __name__ == "__main__":
    main()
