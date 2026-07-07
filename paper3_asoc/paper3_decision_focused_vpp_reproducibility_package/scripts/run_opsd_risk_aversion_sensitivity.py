from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import run_opsd_decision_focused_policy_search as policy
import run_opsd_vpp_risk_simulator as sim


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RISK_WEIGHTS = [0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]


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


def score_candidates(contexts):
    rows = []
    for coef in product(policy.STD_GRID, policy.STD_GRID, policy.NET_GRID, policy.NET_GRID):
        revenue = policy.evaluate_coef(contexts, coef)
        rows.append(
            {
                "coef": coef,
                "train_mean_revenue": float(np.mean(revenue)),
                "train_cvar_10": sim.cvar(revenue, 0.10),
            }
        )
    return rows


def select_for_risk_weight(candidate_scores, risk_weight):
    best = None
    for row in candidate_scores:
        score = row["train_mean_revenue"] + risk_weight * row["train_cvar_10"]
        if best is None or score > best["score"]:
            best = {
                "coef": row["coef"],
                "score": score,
                "train_mean_revenue": row["train_mean_revenue"],
                "train_cvar_10": row["train_cvar_10"],
            }
    return best


def evaluate_policy_on_contexts(contexts, selected, zone, split, risk_weight):
    rows = []
    label = f"DF risk sweep lambda={risk_weight:.2f}"
    for ctx in contexts:
        charge_score, discharge_score = policy.candidate_scores(
            ctx["mean_price"], ctx["std_price"], ctx["mean_net"], selected["coef"]
        )
        charge, discharge = sim.schedule_two_scores(charge_score, discharge_score)
        row = policy.evaluate_schedule(ctx, charge, discharge, ctx["net_error_mean"], label, zone, split)
        row["risk_weight"] = risk_weight
        rows.append(row)
    return rows


def summarize(daily):
    rows = []
    for (split, zone, risk_weight), g in daily.groupby(["split", "zone", "risk_weight"], sort=True):
        revenue = g["revenue"].to_numpy(float)
        regret = g["regret"].to_numpy(float)
        rows.append(
            {
                "split": split,
                "zone": zone,
                "risk_weight": float(risk_weight),
                "days": int(len(g)),
                "mean_revenue": float(np.mean(revenue)),
                "mean_regret": float(np.mean(regret)),
                "median_regret": float(np.median(regret)),
                "negative_revenue_days": int(g["negative_revenue"].sum()),
                "revenue_p05": float(np.quantile(revenue, 0.05)),
                "cvar_10": sim.cvar(revenue, 0.10),
                "mean_imbalance_penalty": float(g["imbalance_penalty"].mean()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_test(summary):
    test = summary[summary["split"] == "test"].copy()
    out = (
        test.groupby("risk_weight", as_index=False)
        .agg(
            mean_revenue=("mean_revenue", "mean"),
            mean_regret=("mean_regret", "mean"),
            cvar_10=("cvar_10", "mean"),
            negative_revenue_days=("negative_revenue_days", "sum"),
            mean_imbalance_penalty=("mean_imbalance_penalty", "mean"),
        )
        .sort_values("risk_weight")
    )
    out["selection_objective"] = out["mean_revenue"] + out["risk_weight"] * out["cvar_10"]
    return out


def plot_risk_sweep(aggregate):
    plot = aggregate.sort_values("risk_weight").copy()
    width, height = 1850, 1120
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "Risk-aversion sensitivity of decision-focused VPP policy", fill="#172033", font=font(42, True))
    d.text((92, 112), "Policies are selected by mean revenue plus lambda_r times train CVaR10 and evaluated on held-out public days", fill="#5f6b7a", font=font(22))

    left, top, right, bottom = 210, 215, 1660, 850
    rev = plot["mean_revenue"].to_numpy(float)
    cvar = plot["cvar_10"].to_numpy(float)
    xvals = plot["risk_weight"].to_numpy(float)
    xmin, xmax = float(xvals.min()), float(xvals.max())
    ymin = min(float(rev.min()), float(cvar.min())) - 1.5
    ymax = max(float(rev.max()), float(cvar.max())) + 1.5

    def xcoord(x):
        return left + (float(x) - xmin) / (xmax - xmin) * (right - left)

    def ycoord(y):
        return bottom - (float(y) - ymin) / (ymax - ymin) * (bottom - top)

    for i in range(6):
        y = top + i * (bottom - top) / 5
        d.line((left, y, right, y), fill="#e5e9f0", width=1)
        val = ymax - i * (ymax - ymin) / 5
        d.text((92, y - 13), f"{val:.1f}", fill="#64748b", font=font(20))
    for x in xvals:
        px = xcoord(x)
        d.line((px, top, px, bottom), fill="#eef2f7", width=1)
        d.text((px - 24, bottom + 26), f"{x:.2g}", fill="#64748b", font=font(20))

    d.line((left, bottom, right, bottom), fill="#8492a6", width=2)
    d.line((left, top, left, bottom), fill="#8492a6", width=2)

    rev_points = [(xcoord(x), ycoord(y)) for x, y in zip(xvals, rev)]
    cvar_points = [(xcoord(x), ycoord(y)) for x, y in zip(xvals, cvar)]
    if len(rev_points) > 1:
        d.line(rev_points, fill="#2f6f9f", width=6)
        d.line(cvar_points, fill="#e45756", width=6)
    for px, py in rev_points:
        d.ellipse((px - 9, py - 9, px + 9, py + 9), fill="#2f6f9f")
    for px, py in cvar_points:
        d.ellipse((px - 9, py - 9, px + 9, py + 9), fill="#e45756")

    d.rounded_rectangle((1240, 230, 1640, 330), radius=8, outline="#d5dbe5", width=2, fill="#ffffff")
    d.line((1270, 265, 1340, 265), fill="#2f6f9f", width=6)
    d.text((1360, 250), "Mean revenue", fill="#263142", font=font(22, True))
    d.line((1270, 305, 1340, 305), fill="#e45756", width=6)
    d.text((1360, 290), "CVaR10", fill="#263142", font=font(22, True))
    d.text((left + 430, bottom + 78), "Risk-aversion coefficient lambda_r", fill="#526070", font=font(23, True))
    d.text((70, top - 42), "EUR/day proxy", fill="#526070", font=font(22, True))

    out = FIGURES / "paper3_fig8_opsd_risk_aversion_sensitivity.png"
    img.save(out)
    return out


def run():
    days_by_zone = sim.prepare_days()
    daily_rows = []
    coef_rows = []
    for zone, zone_days in days_by_zone.items():
        train, validation, test = policy.split_contexts(zone_days)
        train_for_selection = train + validation
        candidate_scores = score_candidates(train_for_selection)
        for risk_weight in RISK_WEIGHTS:
            selected = select_for_risk_weight(candidate_scores, risk_weight)
            c_std, d_std, c_net, d_net = selected["coef"]
            coef_rows.append(
                {
                    "zone": zone,
                    "risk_weight": risk_weight,
                    "charge_std_coef": c_std,
                    "discharge_std_coef": d_std,
                    "charge_net_coef": c_net,
                    "discharge_net_coef": d_net,
                    "selection_score": selected["score"],
                    "selection_mean_revenue": selected["train_mean_revenue"],
                    "selection_cvar_10": selected["train_cvar_10"],
                    "train_days": len(train_for_selection),
                    "test_days": len(test),
                }
            )
            for split, contexts in [("train", train), ("validation", validation), ("test", test)]:
                daily_rows.extend(evaluate_policy_on_contexts(contexts, selected, zone, split, risk_weight))
    daily = pd.DataFrame(daily_rows)
    summary = summarize(daily)
    coefs = pd.DataFrame(coef_rows)
    aggregate = aggregate_test(summary)
    daily.to_csv(RESULTS / "opsd_risk_aversion_sensitivity_daily.csv", index=False)
    summary.to_csv(RESULTS / "opsd_risk_aversion_sensitivity_summary.csv", index=False)
    coefs.to_csv(RESULTS / "opsd_risk_aversion_sensitivity_coefficients.csv", index=False)
    aggregate.to_csv(RESULTS / "opsd_risk_aversion_sensitivity_test_aggregate.csv", index=False)
    fig = plot_risk_sweep(aggregate)
    return daily, summary, coefs, aggregate, fig


if __name__ == "__main__":
    _, _, _, aggregate, fig = run()
    print(RESULTS / "opsd_risk_aversion_sensitivity_daily.csv")
    print(RESULTS / "opsd_risk_aversion_sensitivity_summary.csv")
    print(RESULTS / "opsd_risk_aversion_sensitivity_coefficients.csv")
    print(RESULTS / "opsd_risk_aversion_sensitivity_test_aggregate.csv")
    print(fig)
    print(aggregate.to_string(index=False))
