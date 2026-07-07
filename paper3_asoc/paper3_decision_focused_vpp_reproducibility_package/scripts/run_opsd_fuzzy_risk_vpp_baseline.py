from __future__ import annotations

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

OUT_DAILY = RESULTS / "opsd_fuzzy_risk_vpp_baseline_daily.csv"
OUT_SUMMARY = RESULTS / "opsd_fuzzy_risk_vpp_baseline_summary.csv"
OUT_AGG = RESULTS / "opsd_fuzzy_risk_vpp_baseline_test_aggregate.csv"
OUT_FIG = FIGURES / "paper3_fig11_opsd_fuzzy_risk_vpp_baseline.png"


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


def minmax(values):
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-9:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def triangular(x, center, width):
    return np.maximum(0.0, 1.0 - np.abs(np.asarray(x, dtype=float) - center) / width)


def fuzzy_scores(mean_price, std_price, mean_net):
    """Fuzzy risk-aware scoring using only rolling historical profiles.

    The rule base is intentionally transparent: charge in low-price, low-risk,
    non-peak-net-load hours; discharge in high-price, low-risk, peak-net-load
    hours. High volatility reduces both memberships to avoid aggressive
    schedules when the historical price profile is unstable.
    """
    price = minmax(mean_price)
    volatility = minmax(std_price)
    net = minmax(mean_net)

    low_price = 1.0 - price
    high_price = price
    medium_price = triangular(price, 0.5, 0.5)
    low_vol = 1.0 - volatility
    high_vol = volatility
    low_net = 1.0 - net
    high_net = net

    charge_membership = (
        0.52 * np.minimum(low_price, low_vol)
        + 0.23 * np.minimum(low_price, low_net)
        + 0.15 * np.minimum(medium_price, low_vol)
        - 0.20 * high_vol
    )
    discharge_membership = (
        0.52 * np.minimum(high_price, low_vol)
        + 0.23 * np.minimum(high_price, high_net)
        + 0.15 * np.minimum(medium_price, high_net)
        - 0.20 * high_vol
    )

    # schedule_two_scores selects the lowest charge score and highest discharge
    # score, hence the negative charge membership.
    return -charge_membership, discharge_membership


def evaluate_schedule(ctx, charge_idx, discharge_idx, net_error, method, zone, split):
    revenue, battery_revenue, flex_value, penalty = sim.daily_revenue(
        ctx["price"], charge_idx, discharge_idx, net_error
    )
    return {
        "dataset": "OPSD",
        "zone": zone,
        "date": ctx["date"],
        "split": split,
        "method": method,
        "revenue": revenue,
        "hindsight_revenue": ctx["hindsight"],
        "regret": ctx["hindsight"] - revenue,
        "battery_revenue": battery_revenue,
        "flex_value": flex_value,
        "imbalance_penalty": penalty,
        "negative_revenue": int(revenue < 0),
    }


def fuzzy_rows(ctx, zone, split):
    charge_score, discharge_score = fuzzy_scores(
        ctx["mean_price"], ctx["std_price"], ctx["mean_net"]
    )
    charge, discharge = sim.schedule_two_scores(charge_score, discharge_score)
    return evaluate_schedule(
        ctx,
        charge,
        discharge,
        ctx["net_error_mean"],
        "Fuzzy risk-aware FTO",
        zone,
        split,
    )


def summarize(daily):
    rows = []
    for (split, zone, method), g in daily.groupby(["split", "zone", "method"], sort=True):
        revenue = g["revenue"].to_numpy(float)
        rows.append(
            {
                "split": split,
                "zone": zone,
                "method": method,
                "days": int(len(g)),
                "mean_revenue": float(np.mean(revenue)),
                "mean_regret": float(g["regret"].mean()),
                "median_regret": float(g["regret"].median()),
                "negative_revenue_days": int(g["negative_revenue"].sum()),
                "revenue_p05": float(np.quantile(revenue, 0.05)),
                "cvar_10": sim.cvar(revenue, 0.10),
                "mean_imbalance_penalty": float(g["imbalance_penalty"].mean()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_test(summary):
    test = summary[summary["split"] == "test"].copy()
    return (
        test.groupby("method", as_index=False)
        .agg(
            mean_revenue=("mean_revenue", "mean"),
            mean_regret=("mean_regret", "mean"),
            cvar_10=("cvar_10", "mean"),
            negative_revenue_days=("negative_revenue_days", "sum"),
            mean_imbalance_penalty=("mean_imbalance_penalty", "mean"),
        )
        .sort_values("mean_revenue", ascending=False)
    )


def build_daily():
    rows = []
    for zone, zone_days in sim.prepare_days().items():
        train, validation, test = policy.split_contexts(zone_days)
        for split, contexts in [("train", train), ("validation", validation), ("test", test)]:
            for ctx in contexts:
                rows.extend(policy.baseline_rows(ctx, zone, split))
                rows.append(fuzzy_rows(ctx, zone, split))
    return pd.DataFrame(rows)


def plot(aggregate):
    plot_df = aggregate[aggregate["method"] != "Hindsight optimum"].copy()
    order = [
        "Fuzzy risk-aware FTO",
        "Robust quantile FTO",
        "Rolling-28d mean FTO",
        "Prev-day FTO",
    ]
    plot_df["order"] = plot_df["method"].map({m: i for i, m in enumerate(order)}).fillna(99)
    plot_df = plot_df.sort_values("order")

    width, height = 1900, 1050
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "Fuzzy risk-aware VPP policy baseline", fill="#172033", font=font(44, True))
    d.text(
        (92, 112),
        "Held-out OPSD test split; transparent fuzzy memberships over rolling price, volatility, and net-load shape",
        fill="#5f6b7a",
        font=font(23),
    )
    left, top, right, bottom = 575, 210, 1630, 835
    max_value = max(float(plot_df["mean_revenue"].max()) * 1.18, 1.0)
    for i in range(6):
        x = left + i * (right - left) / 5
        d.line((x, top, x, bottom), fill="#e5e9f0", width=1)
        d.text((x - 16, bottom + 20), f"{max_value * i / 5:.0f}", fill="#6b7280", font=font(21))
    colors = {
        "Fuzzy risk-aware FTO": "#8E6BBE",
        "Robust quantile FTO": "#E45756",
        "Rolling-28d mean FTO": "#4C78A8",
        "Prev-day FTO": "#72B7B2",
    }
    row_h = 128
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        y = top + 35 + idx * row_h
        d.text((90, y + 15), row.method, fill="#263142", font=font(25, True))
        bar_len = (row.mean_revenue / max_value) * (right - left)
        d.rounded_rectangle((left, y, left + bar_len, y + 62), radius=8, fill=colors.get(row.method, "#4C78A8"))
        d.text((left + bar_len + 12, y + 15), f"{row.mean_revenue:.2f}", fill="#263142", font=font(24, True))
        d.text(
            (left + 465, y + 70),
            f"regret {row.mean_regret:.2f}; CVaR10 {row.cvar_10:.2f}; loss days {int(row.negative_revenue_days)}",
            fill="#5f6b7a",
            font=font(20),
        )
    d.text((left, bottom + 70), "Mean daily revenue, higher is better (EUR/day proxy)", fill="#526070", font=font(23, True))
    img.save(OUT_FIG)


def run():
    daily = build_daily()
    summary = summarize(daily)
    aggregate = aggregate_test(summary)
    daily.to_csv(OUT_DAILY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    aggregate.to_csv(OUT_AGG, index=False)
    plot(aggregate)
    return daily, summary, aggregate


def main():
    daily, summary, aggregate = run()
    print(OUT_DAILY)
    print(OUT_SUMMARY)
    print(OUT_AGG)
    print(OUT_FIG)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
