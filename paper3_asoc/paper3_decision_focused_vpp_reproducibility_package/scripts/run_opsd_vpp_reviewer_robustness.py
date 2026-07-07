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

COARSE_COEFS = RESULTS / "opsd_decision_focused_policy_coefficients.csv"

PENALTY_RATES = [20.0, 45.0, 70.0]
TRANSACTION_COSTS = [0.0, 1.0, 3.0]
LOCAL_STD_OFFSETS = [-0.25, 0.0, 0.25]
LOCAL_NET_OFFSETS = [-2.0, 0.0, 2.0]


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


def daily_revenue_param(
    realized_price,
    charge_idx,
    discharge_idx,
    netload_error_ratio,
    penalty_rate=sim.IMBALANCE_PENALTY_RATE,
    transaction_cost=0.0,
    capacity=sim.BATTERY_CAPACITY_MWH,
):
    realized_price = np.asarray(realized_price, dtype=float)
    charge_idx = np.asarray(charge_idx, dtype=int)
    discharge_idx = np.asarray(discharge_idx, dtype=int)
    battery_power = capacity / sim.ACTIVE_HOURS
    flex_power = sim.FLEX_LOAD_MWH / sim.ACTIVE_HOURS
    battery_revenue = battery_power * (
        realized_price[discharge_idx].sum() * sim.EFFICIENCY
        - realized_price[charge_idx].sum() / sim.EFFICIENCY
    )
    flex_value = flex_power * (
        realized_price[discharge_idx].sum() - realized_price[charge_idx].sum()
    )
    active_idx = np.concatenate([charge_idx, discharge_idx])
    imbalance_penalty = float(
        penalty_rate
        * (battery_power + flex_power)
        * np.mean(np.abs(netload_error_ratio[active_idx]))
    )
    active_mwh = (battery_power + flex_power) * len(active_idx)
    transaction_penalty = float(transaction_cost * active_mwh)
    revenue = float(battery_revenue + flex_value - imbalance_penalty - transaction_penalty)
    return revenue, float(battery_revenue), float(flex_value), imbalance_penalty, transaction_penalty


def coef_tuple(row):
    return (
        float(row["charge_std_coef"]),
        float(row["discharge_std_coef"]),
        float(row["charge_net_coef"]),
        float(row["discharge_net_coef"]),
    )


def load_coarse_policies():
    if not COARSE_COEFS.exists():
        raise SystemExit(f"Missing coarse policy coefficients: {COARSE_COEFS}")
    frame = pd.read_csv(COARSE_COEFS)
    out = {}
    for _, row in frame.iterrows():
        out[(row["zone"], row["method"])] = coef_tuple(row)
    return out


def contexts_by_zone():
    days_by_zone = sim.prepare_days()
    out = {}
    for zone, zone_days in days_by_zone.items():
        train, validation, test = policy.split_contexts(zone_days)
        out[zone] = {
            "train_selection": train + validation,
            "test": test,
        }
    return out


def schedule_for_coef(ctx, coef):
    charge_score, discharge_score = policy.candidate_scores(
        ctx["mean_price"], ctx["std_price"], ctx["mean_net"], coef
    )
    return sim.schedule_two_scores(charge_score, discharge_score)


def score_coef(contexts, coef, risk_weight=0.0):
    revenues = []
    for ctx in contexts:
        charge, discharge = schedule_for_coef(ctx, coef)
        revenue, _, _, _, _ = daily_revenue_param(
            ctx["price"], charge, discharge, ctx["net_error_mean"]
        )
        revenues.append(revenue)
    values = np.asarray(revenues, dtype=float)
    return float(np.mean(values) + risk_weight * sim.cvar(values, 0.10)), float(np.mean(values)), sim.cvar(values, 0.10)


def local_refinement_candidates(coef):
    c_std, d_std, c_net, d_net = coef
    seen = set()
    for offsets in product(LOCAL_STD_OFFSETS, LOCAL_STD_OFFSETS, LOCAL_NET_OFFSETS, LOCAL_NET_OFFSETS):
        candidate = (
            round(c_std + offsets[0], 3),
            round(d_std + offsets[1], 3),
            round(c_net + offsets[2], 3),
            round(d_net + offsets[3], 3),
        )
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def select_local_refinement(contexts, base_coef, risk_weight):
    best = None
    for coef in local_refinement_candidates(base_coef):
        score, mean_revenue, cvar_10 = score_coef(contexts, coef, risk_weight=risk_weight)
        if best is None or score > best["score"]:
            best = {
                "coef": coef,
                "score": score,
                "selection_mean_revenue": mean_revenue,
                "selection_cvar_10": cvar_10,
            }
    return best


def stress_rows_for_context(ctx, zone, method, charge, discharge, net_error, penalty_rate, transaction_cost):
    h_revenue, _, _, _, h_transaction = daily_revenue_param(
        ctx["price"], ctx["h_charge"], ctx["h_discharge"], np.zeros_like(ctx["price"]),
        penalty_rate=penalty_rate, transaction_cost=transaction_cost
    )
    revenue, battery, flex, imbalance, transaction = daily_revenue_param(
        ctx["price"], charge, discharge, net_error,
        penalty_rate=penalty_rate, transaction_cost=transaction_cost
    )
    return {
        "dataset": "OPSD",
        "zone": zone,
        "date": ctx["date"],
        "method": method,
        "penalty_rate": penalty_rate,
        "transaction_cost": transaction_cost,
        "revenue": revenue,
        "hindsight_revenue": h_revenue,
        "regret": h_revenue - revenue,
        "battery_revenue": battery,
        "flex_value": flex,
        "imbalance_penalty": imbalance,
        "transaction_penalty": transaction,
        "hindsight_transaction_penalty": h_transaction,
        "negative_revenue": int(revenue < 0),
    }


def evaluate_stress(contexts, coarse):
    rows = []
    for zone, splits in contexts.items():
        test = splits["test"]
        rev_coef = coarse[(zone, "DF policy search (revenue)")]
        risk_coef = coarse[(zone, "DF policy search (risk-adjusted)")]
        for ctx in test:
            mean_charge, mean_discharge = sim.schedule_same_score(ctx["mean_price"])
            robust_charge_score = ctx["mean_price"] + sim.ROBUST_GAMMA * ctx["std_price"]
            robust_discharge_score = ctx["mean_price"] - sim.ROBUST_GAMMA * ctx["std_price"]
            robust_charge, robust_discharge = sim.schedule_two_scores(robust_charge_score, robust_discharge_score)
            rev_charge, rev_discharge = schedule_for_coef(ctx, rev_coef)
            risk_charge, risk_discharge = schedule_for_coef(ctx, risk_coef)
            schedules = [
                ("Rolling-28d mean FTO", mean_charge, mean_discharge, ctx["net_error_mean"]),
                ("Robust quantile FTO", robust_charge, robust_discharge, ctx["net_error_mean"]),
                ("DF policy search (revenue)", rev_charge, rev_discharge, ctx["net_error_mean"]),
                ("DF policy search (risk-adjusted)", risk_charge, risk_discharge, ctx["net_error_mean"]),
            ]
            for penalty_rate in PENALTY_RATES:
                for transaction_cost in TRANSACTION_COSTS:
                    for method, charge, discharge, net_error in schedules:
                        rows.append(
                            stress_rows_for_context(
                                ctx, zone, method, charge, discharge, net_error,
                                penalty_rate, transaction_cost
                            )
                        )
    return pd.DataFrame(rows)


def summarize_stress(daily):
    zone_rows = []
    for (zone, method, penalty_rate, transaction_cost), g in daily.groupby(["zone", "method", "penalty_rate", "transaction_cost"], sort=True):
        revenue = g["revenue"].to_numpy(float)
        zone_rows.append(
            {
                "zone": zone,
                "method": method,
                "penalty_rate": float(penalty_rate),
                "transaction_cost": float(transaction_cost),
                "mean_revenue": float(np.mean(revenue)),
                "mean_regret": float(g["regret"].mean()),
                "cvar_10": sim.cvar(revenue, 0.10),
                "negative_revenue_days": int(g["negative_revenue"].sum()),
                "mean_imbalance_penalty": float(g["imbalance_penalty"].mean()),
                "mean_transaction_penalty": float(g["transaction_penalty"].mean()),
            }
        )
    zone_summary = pd.DataFrame(zone_rows)
    out = (
        zone_summary.groupby(["method", "penalty_rate", "transaction_cost"], as_index=False)
        .agg(
            mean_revenue=("mean_revenue", "mean"),
            mean_regret=("mean_regret", "mean"),
            cvar_10=("cvar_10", "mean"),
            negative_revenue_days=("negative_revenue_days", "sum"),
            mean_imbalance_penalty=("mean_imbalance_penalty", "mean"),
            mean_transaction_penalty=("mean_transaction_penalty", "mean"),
        )
    )
    return out


def aggregate_stress(summary):
    base = summary[(summary["penalty_rate"] == 45.0) & (summary["transaction_cost"] == 0.0)].copy()
    high_cost = summary[(summary["penalty_rate"] == 70.0) & (summary["transaction_cost"] == 3.0)].copy()
    base["scenario"] = "Base settlement"
    high_cost["scenario"] = "High penalty + transaction cost"
    out = pd.concat([base, high_cost], ignore_index=True)
    order = {
        "DF policy search (revenue)": 0,
        "DF policy search (risk-adjusted)": 1,
        "Rolling-28d mean FTO": 2,
        "Robust quantile FTO": 3,
    }
    out["order"] = out["method"].map(order)
    return out.sort_values(["scenario", "order"]).drop(columns=["order"])


def evaluate_fine_grid(contexts, coarse):
    rows = []
    coef_rows = []
    for zone, splits in contexts.items():
        selection = splits["train_selection"]
        test = splits["test"]
        base_revenue = coarse[(zone, "DF policy search (revenue)")]
        base_risk = coarse[(zone, "DF policy search (risk-adjusted)")]
        variants = {
            "Coarse revenue grid": {"coef": base_revenue, "risk_weight": 0.0},
            "Fine revenue grid": {**select_local_refinement(selection, base_revenue, 0.0), "risk_weight": 0.0},
            "Coarse risk-adjusted grid": {"coef": base_risk, "risk_weight": policy.RISK_WEIGHT},
            "Fine risk-adjusted grid": {**select_local_refinement(selection, base_risk, policy.RISK_WEIGHT), "risk_weight": policy.RISK_WEIGHT},
        }
        for label, selected in variants.items():
            coef = selected["coef"]
            coef_rows.append(
                {
                    "zone": zone,
                    "variant": label,
                    "risk_weight": selected["risk_weight"],
                    "charge_std_coef": coef[0],
                    "discharge_std_coef": coef[1],
                    "charge_net_coef": coef[2],
                    "discharge_net_coef": coef[3],
                }
            )
            for ctx in test:
                charge, discharge = schedule_for_coef(ctx, coef)
                row = stress_rows_for_context(
                    ctx, zone, label, charge, discharge, ctx["net_error_mean"],
                    penalty_rate=45.0, transaction_cost=0.0
                )
                rows.append(row)
    daily = pd.DataFrame(rows)
    coefs = pd.DataFrame(coef_rows)
    zone_summary = (
        daily.groupby(["zone", "method"], as_index=False)
        .agg(
            mean_revenue=("revenue", "mean"),
            mean_regret=("regret", "mean"),
            cvar_10=("revenue", lambda x: sim.cvar(x.to_numpy(float), 0.10)),
            negative_revenue_days=("negative_revenue", "sum"),
            mean_imbalance_penalty=("imbalance_penalty", "mean"),
        )
    )
    summary = (
        zone_summary.groupby("method", as_index=False)
        .agg(
            mean_revenue=("mean_revenue", "mean"),
            mean_regret=("mean_regret", "mean"),
            cvar_10=("cvar_10", "mean"),
            negative_revenue_days=("negative_revenue_days", "sum"),
            mean_imbalance_penalty=("mean_imbalance_penalty", "mean"),
        )
    )
    order = {
        "Coarse revenue grid": 0,
        "Fine revenue grid": 1,
        "Coarse risk-adjusted grid": 2,
        "Fine risk-adjusted grid": 3,
    }
    summary["order"] = summary["method"].map(order)
    return daily, summary.sort_values("order").drop(columns=["order"]), coefs


def plot_robustness(stress_aggregate, fine_summary):
    width, height = 1850, 1200
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "Reviewer robustness checks for OPSD VPP decision policies", fill="#172033", font=font(42, True))
    d.text((92, 112), "Stress settlement assumptions and fine policy-grid refinement on held-out public days", fill="#5f6b7a", font=font(22))
    left, top, right = 620, 210, 1640
    colors = {
        "DF policy search (revenue)": "#2f6f9f",
        "DF policy search (risk-adjusted)": "#e45756",
        "Rolling-28d mean FTO": "#4f7cac",
        "Robust quantile FTO": "#72b7b2",
        "Coarse revenue grid": "#2f6f9f",
        "Fine revenue grid": "#9ecae1",
        "Coarse risk-adjusted grid": "#e45756",
        "Fine risk-adjusted grid": "#f4a6a6",
    }

    panel = stress_aggregate[stress_aggregate["scenario"] == "High penalty + transaction cost"].copy()
    panel = panel.sort_values("mean_revenue", ascending=True)
    max_v = max(float(panel["mean_revenue"].max()) * 1.18, 1.0)
    d.text((90, 178), "A. High penalty + transaction-cost stress", fill="#263142", font=font(30, True))
    for i, row in enumerate(panel.itertuples(index=False)):
        y = top + i * 78
        d.text((90, y + 14), row.method, fill="#263142", font=font(22, True))
        x1 = left + row.mean_revenue / max_v * (right - left)
        d.rounded_rectangle((left, y, x1, y + 42), radius=6, fill=colors.get(row.method, "#4f7cac"))
        d.text((x1 + 12, y + 8), f"{row.mean_revenue:.2f}", fill="#263142", font=font(21, True))
        d.text((left + 530, y + 47), f"CVaR10 {row.cvar_10:.2f}; loss days {int(row.negative_revenue_days)}", fill="#64748b", font=font(18))

    d.text((90, 610), "B. Coarse versus fine policy grid", fill="#263142", font=font(30, True))
    fine = fine_summary.sort_values("mean_revenue", ascending=True)
    max_f = max(float(fine["mean_revenue"].max()) * 1.18, 1.0)
    for i, row in enumerate(fine.itertuples(index=False)):
        y = 660 + i * 78
        d.text((90, y + 14), row.method, fill="#263142", font=font(22, True))
        x1 = left + row.mean_revenue / max_f * (right - left)
        d.rounded_rectangle((left, y, x1, y + 42), radius=6, fill=colors.get(row.method, "#4f7cac"))
        d.text((x1 + 12, y + 8), f"{row.mean_revenue:.2f}", fill="#263142", font=font(21, True))
        d.text((left + 530, y + 47), f"CVaR10 {row.cvar_10:.2f}; loss days {int(row.negative_revenue_days)}", fill="#64748b", font=font(18))
    d.text((left, 1080), "Mean daily revenue on final 20% test split (EUR/day proxy)", fill="#526070", font=font(22, True))
    out = FIGURES / "paper3_fig9_opsd_reviewer_robustness.png"
    img.save(out)
    return out


def run():
    contexts = contexts_by_zone()
    coarse = load_coarse_policies()
    stress_daily = evaluate_stress(contexts, coarse)
    stress_summary = summarize_stress(stress_daily)
    stress_aggregate = aggregate_stress(stress_summary)
    fine_daily, fine_summary, fine_coefs = evaluate_fine_grid(contexts, coarse)
    fig = plot_robustness(stress_aggregate, fine_summary)
    stress_daily.to_csv(RESULTS / "opsd_vpp_reviewer_robustness_stress_daily.csv", index=False)
    stress_summary.to_csv(RESULTS / "opsd_vpp_reviewer_robustness_stress_summary.csv", index=False)
    stress_aggregate.to_csv(RESULTS / "opsd_vpp_reviewer_robustness_stress_aggregate.csv", index=False)
    fine_daily.to_csv(RESULTS / "opsd_vpp_reviewer_robustness_fine_grid_daily.csv", index=False)
    fine_summary.to_csv(RESULTS / "opsd_vpp_reviewer_robustness_fine_grid_summary.csv", index=False)
    fine_coefs.to_csv(RESULTS / "opsd_vpp_reviewer_robustness_fine_grid_coefficients.csv", index=False)
    return stress_aggregate, fine_summary, fig


if __name__ == "__main__":
    stress_aggregate, fine_summary, fig = run()
    print(RESULTS / "opsd_vpp_reviewer_robustness_stress_aggregate.csv")
    print(RESULTS / "opsd_vpp_reviewer_robustness_fine_grid_summary.csv")
    print(fig)
    print(stress_aggregate.to_string(index=False))
    print(fine_summary.to_string(index=False))
