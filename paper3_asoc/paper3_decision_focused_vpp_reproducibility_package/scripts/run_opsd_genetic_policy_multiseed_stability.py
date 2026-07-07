from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import run_opsd_genetic_policy_search_vpp_baseline as genetic
import run_opsd_vpp_risk_simulator as sim
import run_opsd_decision_focused_policy_search as policy


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

OUT_BY_SEED = RESULTS / "opsd_genetic_policy_multiseed_stability_by_seed.csv"
OUT_BY_ZONE = RESULTS / "opsd_genetic_policy_multiseed_stability_by_zone.csv"
OUT_COEFS = RESULTS / "opsd_genetic_policy_multiseed_stability_coefficients.csv"
OUT_SUMMARY = RESULTS / "opsd_genetic_policy_multiseed_stability_summary.csv"
OUT_FIG = FIGURES / "paper3_fig15_opsd_genetic_multiseed_stability.png"

N_SEEDS = 8
OBJECTIVE = "risk_adjusted"


def font(size=28, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def evaluate_test_revenues(test_contexts: list[dict], selected: dict, zone: str) -> np.ndarray:
    revenues = []
    for ctx in test_contexts:
        charge_score, discharge_score = policy.candidate_scores(
            ctx["mean_price"], ctx["std_price"], ctx["mean_net"], tuple(selected["coef"])
        )
        charge, discharge = sim.schedule_two_scores(charge_score, discharge_score)
        row = policy.evaluate_schedule(
            ctx, charge, discharge, ctx["net_error_mean"], "Genetic multiseed risk-adjusted", zone, "test"
        )
        revenues.append(float(row["revenue"]))
    return np.asarray(revenues, dtype=float)


def baseline_values() -> dict:
    if not genetic.OUT_AGG.exists():
        genetic.run()
    agg = pd.read_csv(genetic.OUT_AGG)
    out = {}
    for method in [
        "Genetic policy search (risk-adjusted)",
        "DF policy search (revenue)",
        "DF policy search (risk-adjusted)",
        "Robust quantile FTO",
    ]:
        row = agg[agg["method"] == method]
        if not row.empty:
            out[method] = {
                "mean_revenue": float(row.iloc[0]["mean_revenue"]),
                "cvar_10": float(row.iloc[0]["cvar_10"]),
                "negative_revenue_days": int(row.iloc[0]["negative_revenue_days"]),
            }
    return out


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days_by_zone = sim.prepare_days()
    zone_rows = []
    coef_rows = []
    for seed_idx in range(N_SEEDS):
        for zone_idx, (zone, zone_days) in enumerate(days_by_zone.items()):
            train, validation, test = policy.split_contexts(zone_days)
            selection_contexts = genetic.deterministic_selection_subset(train + validation)
            selected = genetic.select_genetic_policy(
                selection_contexts,
                OBJECTIVE,
                seed_offset=1000 + seed_idx * 100 + zone_idx,
            )
            revenues = evaluate_test_revenues(test, selected, zone)
            zone_rows.append(
                {
                    "seed_index": seed_idx,
                    "zone": zone,
                    "test_days": int(len(revenues)),
                    "mean_revenue": float(np.mean(revenues)),
                    "cvar_10": sim.cvar(revenues, 0.10),
                    "negative_revenue_days": int(np.sum(revenues < 0)),
                    "selection_score": float(selected["score"]),
                    "selection_mean_revenue": float(selected["train_mean_revenue"]),
                    "selection_cvar_10": float(selected["train_cvar_10"]),
                    "best_generation": int(selected["generation"]),
                }
            )
            c_std, d_std, c_net, d_net = [float(x) for x in selected["coef"]]
            coef_rows.append(
                {
                    "seed_index": seed_idx,
                    "zone": zone,
                    "charge_std_coef": c_std,
                    "discharge_std_coef": d_std,
                    "charge_net_coef": c_net,
                    "discharge_net_coef": d_net,
                    "best_generation": int(selected["generation"]),
                }
            )

    by_zone = pd.DataFrame(zone_rows)
    by_seed = (
        by_zone.groupby("seed_index", as_index=False)
        .agg(
            mean_revenue=("mean_revenue", "mean"),
            cvar_10=("cvar_10", "mean"),
            negative_revenue_days=("negative_revenue_days", "sum"),
            zones=("zone", "nunique"),
            test_days=("test_days", "sum"),
        )
        .sort_values("seed_index")
    )
    coefs = pd.DataFrame(coef_rows)
    baselines = baseline_values()
    baseline_revenue = baselines.get("DF policy search (revenue)", {}).get("mean_revenue", np.nan)
    baseline_cvar = baselines.get("DF policy search (revenue)", {}).get("cvar_10", np.nan)
    summary = pd.DataFrame(
        [
            {
                "metric": "seed_count",
                "value": float(len(by_seed)),
                "interpretation": "Number of independent genetic-search seeds.",
            },
            {
                "metric": "mean_revenue_mean",
                "value": float(by_seed["mean_revenue"].mean()),
                "interpretation": "Average of seed-level mean test revenues.",
            },
            {
                "metric": "mean_revenue_std",
                "value": float(by_seed["mean_revenue"].std(ddof=1)),
                "interpretation": "Seed-level standard deviation of mean test revenue.",
            },
            {
                "metric": "mean_revenue_min",
                "value": float(by_seed["mean_revenue"].min()),
                "interpretation": "Worst seed-level mean test revenue.",
            },
            {
                "metric": "mean_revenue_max",
                "value": float(by_seed["mean_revenue"].max()),
                "interpretation": "Best seed-level mean test revenue.",
            },
            {
                "metric": "cvar_10_mean",
                "value": float(by_seed["cvar_10"].mean()),
                "interpretation": "Average seed-level CVaR10.",
            },
            {
                "metric": "cvar_10_std",
                "value": float(by_seed["cvar_10"].std(ddof=1)),
                "interpretation": "Seed-level standard deviation of CVaR10.",
            },
            {
                "metric": "seeds_at_or_above_df_revenue",
                "value": float(np.sum(by_seed["mean_revenue"].to_numpy(float) >= baseline_revenue)),
                "interpretation": "Seeds whose mean revenue is at least the coarse-grid revenue-selected policy.",
            },
            {
                "metric": "df_revenue_baseline",
                "value": float(baseline_revenue),
                "interpretation": "Coarse-grid revenue-selected decision-focused policy mean revenue.",
            },
            {
                "metric": "df_revenue_baseline_cvar_10",
                "value": float(baseline_cvar),
                "interpretation": "Coarse-grid revenue-selected decision-focused policy CVaR10.",
            },
        ]
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(OUT_BY_SEED, index=False)
    by_zone.to_csv(OUT_BY_ZONE, index=False)
    coefs.to_csv(OUT_COEFS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    plot(by_seed, baselines)
    return by_seed, by_zone, coefs, summary


def plot(by_seed: pd.DataFrame, baselines: dict) -> None:
    width, height = 1900, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 58), "Multi-seed stability of genetic policy search", fill="#172033", font=font(44, True))
    draw.text(
        (82, 118),
        "Eight independent seeds; coefficients selected on chronological train/validation days and frozen on final OPSD test days",
        fill="#5f6b7a",
        font=font(22),
    )
    values = by_seed["mean_revenue"].to_numpy(float)
    cvars = by_seed["cvar_10"].to_numpy(float)
    df_rev = baselines.get("DF policy search (revenue)", {}).get("mean_revenue", np.nan)
    robust_rev = baselines.get("Robust quantile FTO", {}).get("mean_revenue", np.nan)

    left, top, right, bottom = 230, 280, 1620, 760
    min_v = min(float(np.min(values)) - 0.25, robust_rev - 0.5)
    max_v = max(float(np.max(values)) + 0.25, df_rev + 0.5)
    span = max_v - min_v

    def xcoord(v):
        return left + (float(v) - min_v) / span * (right - left)

    for i in range(6):
        x = left + i * (right - left) / 5
        draw.line((x, top - 28, x, bottom + 35), fill="#e5e9f0", width=1)
        tick = min_v + span * i / 5
        draw.text((x - 36, bottom + 52), f"{tick:.1f}", fill="#64748b", font=font(19))

    y = top + 165
    x_min, x_max = xcoord(np.min(values)), xcoord(np.max(values))
    x_q1, x_q3 = xcoord(np.quantile(values, 0.25)), xcoord(np.quantile(values, 0.75))
    x_med = xcoord(np.median(values))
    draw.line((x_min, y, x_max, y), fill="#0f766e", width=8)
    draw.rounded_rectangle((x_q1, y - 48, x_q3, y + 48), radius=10, fill="#99f6e4", outline="#0f766e", width=3)
    draw.line((x_med, y - 60, x_med, y + 60), fill="#0f766e", width=6)
    label_offsets = [-18, 18, 54, 90]
    for idx, row in by_seed.iterrows():
        px = xcoord(row.mean_revenue)
        py = y + 125 + label_offsets[int(row.seed_index) % len(label_offsets)]
        draw.ellipse((px - 12, py - 12, px + 12, py + 12), fill="#0f766e")
        draw.text((px - 12, py + 18), str(int(row.seed_index)), fill="#334155", font=font(16))

    if not np.isnan(df_rev):
        x = xcoord(df_rev)
        draw.line((x, top - 35, x, bottom + 20), fill="#ef4444", width=4)
        draw.text((x + 10, top - 65), f"DF grid revenue {df_rev:.2f}", fill="#b91c1c", font=font(20, True))
    if not np.isnan(robust_rev):
        x = xcoord(robust_rev)
        draw.line((x, top - 35, x, bottom + 20), fill="#475569", width=3)
        draw.text((x + 10, top - 96), f"Robust FTO {robust_rev:.2f}", fill="#475569", font=font(19))

    draw.text((left, bottom + 95), "Mean daily revenue on final 20% test split (EUR/day proxy)", fill="#334155", font=font(23, True))
    draw.text((80, 880), f"Revenue across seeds: {values.mean():.3f} +/- {values.std(ddof=1):.3f} EUR/day", fill="#172033", font=font(27, True))
    draw.text((80, 925), f"CVaR10 across seeds: {cvars.mean():.3f} +/- {cvars.std(ddof=1):.3f}; range {cvars.min():.3f} to {cvars.max():.3f}", fill="#334155", font=font(23))
    if not np.isnan(df_rev):
        count = int(np.sum(values >= df_rev))
        draw.text((80, 970), f"{count}/{len(values)} seeds match or exceed the coarse-grid revenue-selected policy.", fill="#334155", font=font(23))
    draw.text(
        (80, 1030),
        "Interpretation boundary: stability supports the evolutionary search as a reproducible soft-computing refinement, not as a universal optimizer.",
        fill="#64748b",
        font=font(20),
    )
    image.save(OUT_FIG)


def main() -> None:
    by_seed, by_zone, coefs, summary = run()
    print(OUT_BY_SEED)
    print(OUT_BY_ZONE)
    print(OUT_COEFS)
    print(OUT_SUMMARY)
    print(OUT_FIG)
    print(by_seed.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
