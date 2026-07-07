from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import run_opsd_vpp_risk_simulator as sim


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

STD_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]
NET_GRID = [-8.0, -4.0, 0.0, 4.0, 8.0]
RISK_WEIGHT = 0.50


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


def net_shape(mean_net):
    values = np.asarray(mean_net, dtype=float)
    std = float(values.std())
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def candidate_scores(mean_price, std_price, mean_net, coef):
    c_std, d_std, c_net, d_net = coef
    shape = net_shape(mean_net)
    charge_score = mean_price + c_std * std_price + c_net * shape
    discharge_score = mean_price + d_std * std_price + d_net * shape
    return charge_score, discharge_score


def day_context(zone_days, idx):
    day = zone_days[idx]
    prev_price, prev_net, mean_price, std_price, mean_net = sim.historical_profiles(zone_days, idx)
    scale = day["scale"]
    price = day["price"]
    net_load = day["net_load"]
    net_error_prev = (net_load - prev_net) / scale
    net_error_mean = (net_load - mean_net) / scale
    h_charge, h_discharge = sim.schedule_same_score(price)
    hindsight, _, _, _ = sim.daily_revenue(price, h_charge, h_discharge, np.zeros_like(price))
    return {
        "date": day["date"].date().isoformat(),
        "price": price,
        "prev_price": prev_price,
        "mean_price": mean_price,
        "std_price": std_price,
        "mean_net": mean_net,
        "net_error_prev": net_error_prev,
        "net_error_mean": net_error_mean,
        "hindsight": hindsight,
        "h_charge": h_charge,
        "h_discharge": h_discharge,
    }


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


def evaluate_coef(contexts, coef):
    revenues = []
    for ctx in contexts:
        charge_score, discharge_score = candidate_scores(
            ctx["mean_price"], ctx["std_price"], ctx["mean_net"], coef
        )
        charge, discharge = sim.schedule_two_scores(charge_score, discharge_score)
        revenue, _, _, _ = sim.daily_revenue(ctx["price"], charge, discharge, ctx["net_error_mean"])
        revenues.append(revenue)
    return np.asarray(revenues, dtype=float)


def select_policy(contexts, objective):
    best = None
    for coef in product(STD_GRID, STD_GRID, NET_GRID, NET_GRID):
        revenue = evaluate_coef(contexts, coef)
        if objective == "revenue":
            score = float(np.mean(revenue))
        elif objective == "risk_adjusted":
            score = float(np.mean(revenue) + RISK_WEIGHT * sim.cvar(revenue, 0.10))
        else:
            raise ValueError(objective)
        if best is None or score > best["score"]:
            best = {
                "coef": coef,
                "score": score,
                "train_mean_revenue": float(np.mean(revenue)),
                "train_cvar_10": sim.cvar(revenue, 0.10),
            }
    return best


def split_contexts(zone_days):
    start_idx = max(sim.ROLLING_WINDOW_DAYS, 1)
    contexts = [day_context(zone_days, idx) for idx in range(start_idx, len(zone_days))]
    n = len(contexts)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    return contexts[:train_end], contexts[train_end:validation_end], contexts[validation_end:]


def baseline_rows(ctx, zone, split):
    rows = []
    prev_charge, prev_discharge = sim.schedule_same_score(ctx["prev_price"])
    mean_charge, mean_discharge = sim.schedule_same_score(ctx["mean_price"])
    robust_charge_score = ctx["mean_price"] + sim.ROBUST_GAMMA * ctx["std_price"]
    robust_discharge_score = ctx["mean_price"] - sim.ROBUST_GAMMA * ctx["std_price"]
    robust_charge, robust_discharge = sim.schedule_two_scores(robust_charge_score, robust_discharge_score)

    rows.append(evaluate_schedule(ctx, ctx["h_charge"], ctx["h_discharge"], np.zeros_like(ctx["price"]), "Hindsight optimum", zone, split))
    rows.append(evaluate_schedule(ctx, prev_charge, prev_discharge, ctx["net_error_prev"], "Prev-day FTO", zone, split))
    rows.append(evaluate_schedule(ctx, mean_charge, mean_discharge, ctx["net_error_mean"], "Rolling-28d mean FTO", zone, split))
    rows.append(evaluate_schedule(ctx, robust_charge, robust_discharge, ctx["net_error_mean"], "Robust quantile FTO", zone, split))
    return rows


def policy_rows(ctx, zone, split, policies):
    rows = []
    for label, selected in policies.items():
        charge_score, discharge_score = candidate_scores(
            ctx["mean_price"], ctx["std_price"], ctx["mean_net"], selected["coef"]
        )
        charge, discharge = sim.schedule_two_scores(charge_score, discharge_score)
        rows.append(evaluate_schedule(ctx, charge, discharge, ctx["net_error_mean"], label, zone, split))
    return rows


def summarize(daily):
    rows = []
    for (split, zone, method), g in daily.groupby(["split", "zone", "method"], sort=True):
        revenue = g["revenue"].to_numpy(float)
        regret = g["regret"].to_numpy(float)
        rows.append(
            {
                "split": split,
                "zone": zone,
                "method": method,
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


def plot_decision_policy(summary):
    plot = aggregate_test(summary)
    plot = plot[plot["method"] != "Hindsight optimum"].copy()
    order = [
        "DF policy search (revenue)",
        "DF policy search (risk-adjusted)",
        "Rolling-28d mean FTO",
        "Robust quantile FTO",
        "Prev-day FTO",
    ]
    plot["order"] = plot["method"].map({m: i for i, m in enumerate(order)})
    plot = plot.sort_values("order")
    width, height = 1800, 1100
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "Decision-focused policy search on OPSD test split", fill="#172033", font=font(42, True))
    d.text((92, 112), "Policies are selected on training revenue/risk and evaluated on held-out public days", fill="#5f6b7a", font=font(23))
    left, top, right, bottom = 585, 215, 1620, 870
    values = plot["mean_revenue"].to_numpy(float)
    max_value = max(float(values.max()) * 1.15, 1.0)
    for i in range(6):
        x = left + i * (right - left) / 5
        d.line((x, top, x, bottom), fill="#e5e9f0", width=1)
        label = f"{max_value * i / 5:.0f}"
        d.text((x - 18, bottom + 25), label, fill="#6b7280", font=font(21))
    colors = ["#2f6f9f", "#e45756", "#4f7cac", "#72b7b2", "#9a9a9a"]
    row_h = 110
    for i, row in enumerate(plot.itertuples(index=False)):
        y = top + 35 + i * row_h
        d.text((90, y + 18), row.method, fill="#263142", font=font(24, True))
        bar_len = (row.mean_revenue / max_value) * (right - left)
        d.rounded_rectangle((left, y, left + bar_len, y + 62), radius=8, fill=colors[i % len(colors)])
        d.text((left + bar_len + 12, y + 17), f"{row.mean_revenue:.2f}", fill="#263142", font=font(24, True))
        d.text((left + 500, y + 70), f"regret {row.mean_regret:.2f}; CVaR10 {row.cvar_10:.2f}", fill="#5f6b7a", font=font(20))
    d.text((left, bottom + 74), "Mean daily revenue on final 20% test split (EUR/day proxy)", fill="#526070", font=font(23, True))
    out = FIGURES / "paper3_fig6_opsd_decision_focused_policy_search.png"
    img.save(out)
    return out


def run():
    days_by_zone = sim.prepare_days()
    daily_rows = []
    coef_rows = []
    for zone, zone_days in days_by_zone.items():
        train, validation, test = split_contexts(zone_days)
        train_for_selection = train + validation
        policies = {
            "DF policy search (revenue)": select_policy(train_for_selection, "revenue"),
            "DF policy search (risk-adjusted)": select_policy(train_for_selection, "risk_adjusted"),
        }
        for label, selected in policies.items():
            c_std, d_std, c_net, d_net = selected["coef"]
            coef_rows.append(
                {
                    "zone": zone,
                    "method": label,
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
            for ctx in contexts:
                daily_rows.extend(baseline_rows(ctx, zone, split))
                daily_rows.extend(policy_rows(ctx, zone, split, policies))
    daily = pd.DataFrame(daily_rows)
    summary = summarize(daily)
    coefs = pd.DataFrame(coef_rows)
    aggregate = aggregate_test(summary)
    daily.to_csv(RESULTS / "opsd_decision_focused_policy_daily.csv", index=False)
    summary.to_csv(RESULTS / "opsd_decision_focused_policy_summary.csv", index=False)
    coefs.to_csv(RESULTS / "opsd_decision_focused_policy_coefficients.csv", index=False)
    aggregate.to_csv(RESULTS / "opsd_decision_focused_policy_test_aggregate.csv", index=False)
    fig = plot_decision_policy(summary)
    return daily, summary, coefs, aggregate, fig


if __name__ == "__main__":
    _, _, _, aggregate, fig = run()
    print(RESULTS / "opsd_decision_focused_policy_daily.csv")
    print(RESULTS / "opsd_decision_focused_policy_summary.csv")
    print(RESULTS / "opsd_decision_focused_policy_coefficients.csv")
    print(RESULTS / "opsd_decision_focused_policy_test_aggregate.csv")
    print(fig)
    print(aggregate.to_string(index=False))
