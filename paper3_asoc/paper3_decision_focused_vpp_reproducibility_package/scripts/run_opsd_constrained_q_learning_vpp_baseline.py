from __future__ import annotations

from collections import defaultdict
from math import comb
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

OUT_DAILY = RESULTS / "opsd_constrained_q_learning_vpp_daily.csv"
OUT_SUMMARY = RESULTS / "opsd_constrained_q_learning_vpp_summary.csv"
OUT_AGG = RESULTS / "opsd_constrained_q_learning_vpp_test_aggregate.csv"
OUT_PAIRED = RESULTS / "opsd_constrained_q_learning_vpp_paired_tests.csv"
OUT_FIG = FIGURES / "paper3_fig12_opsd_constrained_q_learning_vpp.png"
DECISION_DAILY = RESULTS / "opsd_decision_focused_policy_daily.csv"

EPISODES = 14
SEED = 20260705
GAMMA = 0.96
ALPHA0 = 0.18
ALPHA_MIN = 0.035
EPS0 = 0.22
EPS_MIN = 0.025


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


def quantile_edges(train_contexts, key, quantiles):
    values = []
    for ctx in train_contexts:
        if key == "net_shape":
            values.extend(net_shape(ctx["mean_net"]))
        else:
            values.extend(np.asarray(ctx[key], dtype=float))
    return np.quantile(np.asarray(values, dtype=float), quantiles)


def build_discretizer(train_contexts):
    return {
        "price": quantile_edges(train_contexts, "mean_price", [0.25, 0.50, 0.75]),
        "volatility": quantile_edges(train_contexts, "std_price", [0.50, 0.80]),
        "net": quantile_edges(train_contexts, "net_shape", [0.25, 0.50, 0.75]),
    }


def bucket(value, edges):
    return int(np.searchsorted(edges, float(value), side="right"))


def state_for(ctx, hour, remaining_charge, remaining_discharge, discretizer):
    hour_block = int(hour // 4)
    if "q_price_bin" in ctx:
        price_bin = int(ctx["q_price_bin"][hour])
        vol_bin = int(ctx["q_vol_bin"][hour])
        net_bin = int(ctx["q_net_bin"][hour])
    else:
        price_bin = bucket(ctx["mean_price"][hour], discretizer["price"])
        vol_bin = bucket(ctx["std_price"][hour], discretizer["volatility"])
        net_bin = bucket(net_shape(ctx["mean_net"])[hour], discretizer["net"])
    return (
        hour_block,
        price_bin,
        vol_bin,
        net_bin,
        int(remaining_charge),
        int(remaining_discharge),
    )


def attach_discrete_state_features(contexts, discretizer):
    out = []
    for ctx in contexts:
        copied = dict(ctx)
        copied["q_price_bin"] = np.asarray([bucket(v, discretizer["price"]) for v in ctx["mean_price"]], dtype=int)
        copied["q_vol_bin"] = np.asarray([bucket(v, discretizer["volatility"]) for v in ctx["std_price"]], dtype=int)
        copied["q_net_bin"] = np.asarray([bucket(v, discretizer["net"]) for v in net_shape(ctx["mean_net"])], dtype=int)
        out.append(copied)
    return out


def valid_actions(hour, remaining_charge, remaining_discharge):
    hours_left = 24 - int(hour)
    actions = [0]
    if remaining_charge > 0:
        actions.append(1)
    if remaining_discharge > 0:
        actions.append(2)
    if remaining_charge + remaining_discharge >= hours_left:
        actions = [a for a in actions if a != 0]
    return actions


def action_reward(ctx, hour, action):
    if action == 0:
        return 0.0
    price = float(ctx["price"][hour])
    battery_power = sim.BATTERY_CAPACITY_MWH / sim.ACTIVE_HOURS
    flex_power = sim.FLEX_LOAD_MWH / sim.ACTIVE_HOURS
    penalty = (
        sim.IMBALANCE_PENALTY_RATE
        * (battery_power + flex_power)
        * abs(float(ctx["net_error_mean"][hour]))
        / (2 * sim.ACTIVE_HOURS)
    )
    if action == 1:
        return -battery_power * price / sim.EFFICIENCY - flex_power * price - penalty
    return battery_power * price * sim.EFFICIENCY + flex_power * price - penalty


def transition_counts(remaining_charge, remaining_discharge, action):
    if action == 1:
        return remaining_charge - 1, remaining_discharge
    if action == 2:
        return remaining_charge, remaining_discharge - 1
    return remaining_charge, remaining_discharge


def choose_action(q_table, state, actions, rng, epsilon):
    if rng.random() < epsilon:
        return int(rng.choice(actions))
    values = np.asarray([q_table[(state, a)] for a in actions], dtype=float)
    max_value = float(values.max())
    best = [a for a, v in zip(actions, values) if abs(float(v) - max_value) < 1e-12]
    return int(best[0])


def train_q_policy(train_contexts, discretizer):
    q_table = defaultdict(float)
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(train_contexts))
    for episode in range(EPISODES):
        rng.shuffle(indices)
        epsilon = max(EPS_MIN, EPS0 * (1.0 - episode / EPISODES))
        alpha = max(ALPHA_MIN, ALPHA0 * (1.0 - episode / EPISODES))
        for idx in indices:
            ctx = train_contexts[int(idx)]
            remaining_charge = sim.ACTIVE_HOURS
            remaining_discharge = sim.ACTIVE_HOURS
            for hour in range(24):
                state = state_for(ctx, hour, remaining_charge, remaining_discharge, discretizer)
                actions = valid_actions(hour, remaining_charge, remaining_discharge)
                action = choose_action(q_table, state, actions, rng, epsilon)
                reward = action_reward(ctx, hour, action)
                next_charge, next_discharge = transition_counts(
                    remaining_charge, remaining_discharge, action
                )
                if hour == 23:
                    future = 0.0
                else:
                    next_state = state_for(ctx, hour + 1, next_charge, next_discharge, discretizer)
                    next_actions = valid_actions(hour + 1, next_charge, next_discharge)
                    future = max(q_table[(next_state, a)] for a in next_actions)
                old = q_table[(state, action)]
                q_table[(state, action)] = old + alpha * (reward + GAMMA * future - old)
                remaining_charge, remaining_discharge = next_charge, next_discharge
    return q_table


def rollout_policy(ctx, q_table, discretizer):
    charge = []
    discharge = []
    remaining_charge = sim.ACTIVE_HOURS
    remaining_discharge = sim.ACTIVE_HOURS
    for hour in range(24):
        state = state_for(ctx, hour, remaining_charge, remaining_discharge, discretizer)
        actions = valid_actions(hour, remaining_charge, remaining_discharge)
        values = np.asarray([q_table[(state, a)] for a in actions], dtype=float)
        action = int(actions[int(np.argmax(values))])
        if action == 1:
            charge.append(hour)
        elif action == 2:
            discharge.append(hour)
        remaining_charge, remaining_discharge = transition_counts(
            remaining_charge, remaining_discharge, action
        )
    if len(charge) != sim.ACTIVE_HOURS or len(discharge) != sim.ACTIVE_HOURS:
        raise RuntimeError(f"Invalid constrained policy rollout: charge={charge}, discharge={discharge}")
    return np.asarray(charge, dtype=int), np.asarray(discharge, dtype=int)


def evaluate_q_contexts(contexts, q_table, discretizer, zone, split):
    rows = []
    for ctx in contexts:
        charge, discharge = rollout_policy(ctx, q_table, discretizer)
        rows.append(
            policy.evaluate_schedule(
                ctx,
                charge,
                discharge,
                ctx["net_error_mean"],
                "Constrained Q-learning policy",
                zone,
                split,
            )
        )
    return rows


def df_policy_rows(ctx, zone, split, policies):
    rows = []
    for label, selected in policies.items():
        charge_score, discharge_score = policy.candidate_scores(
            ctx["mean_price"], ctx["std_price"], ctx["mean_net"], selected["coef"]
        )
        charge, discharge = sim.schedule_two_scores(charge_score, discharge_score)
        rows.append(policy.evaluate_schedule(ctx, charge, discharge, ctx["net_error_mean"], label, zone, split))
    return rows


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


def exact_two_sided_sign_test(wins, losses):
    n = int(wins + losses)
    if n == 0:
        return 1.0
    k = min(int(wins), int(losses))
    prob = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2 * prob))


def paired_tests(daily):
    test = daily[daily["split"] == "test"].copy()
    key_cols = ["zone", "date"]
    methods = [
        "DF policy search (revenue)",
        "DF policy search (risk-adjusted)",
        "Robust quantile FTO",
        "Rolling-28d mean FTO",
        "Fuzzy risk-aware FTO",
    ]
    q = test[test["method"] == "Constrained Q-learning policy"][key_cols + ["revenue"]].rename(
        columns={"revenue": "q_learning_revenue"}
    )
    rows = []
    for method in methods:
        other = test[test["method"] == method][key_cols + ["revenue"]].rename(columns={"revenue": "baseline_revenue"})
        joined = q.merge(other, on=key_cols, how="inner")
        diff = joined["q_learning_revenue"].to_numpy(float) - joined["baseline_revenue"].to_numpy(float)
        wins = int(np.sum(diff > 1e-9))
        losses = int(np.sum(diff < -1e-9))
        ties = int(len(diff) - wins - losses)
        rows.append(
            {
                "comparison": f"Constrained Q-learning policy vs {method}",
                "paired_days": int(len(diff)),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "mean_revenue_delta": float(np.mean(diff)) if len(diff) else np.nan,
                "median_revenue_delta": float(np.median(diff)) if len(diff) else np.nan,
                "exact_two_sided_sign_p": exact_two_sided_sign_test(wins, losses),
            }
        )
    return pd.DataFrame(rows)


def plot(aggregate):
    order = [
        "Constrained Q-learning policy",
        "DF policy search (revenue)",
        "DF policy search (risk-adjusted)",
        "Rolling-28d mean FTO",
        "Robust quantile FTO",
        "Fuzzy risk-aware FTO",
        "Prev-day FTO",
    ]
    plot_df = aggregate[aggregate["method"].isin(order)].copy()
    plot_df["order"] = plot_df["method"].map({m: i for i, m in enumerate(order)})
    plot_df = plot_df.sort_values("order")

    width, height = 1980, 1120
    left, top, right, bottom = 640, 210, 1690, 880
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 55), "Constrained Q-learning VPP decision baseline", fill="#172033", font=font(43, True))
    draw.text(
        (92, 112),
        "Tabular policy is trained on chronological public OPSD days and frozen before final 20% test evaluation",
        fill="#5f6b7a",
        font=font(22),
    )
    min_value = min(0.0, float(plot_df["mean_revenue"].min()) * 1.18)
    max_value = max(float(plot_df["mean_revenue"].max()) * 1.18, 1.0)
    value_span = max_value - min_value

    def xcoord(value):
        return left + (float(value) - min_value) / value_span * (right - left)

    zero_x = xcoord(0.0)
    for i in range(6):
        x = left + i * (right - left) / 5
        draw.line((x, top, x, bottom), fill="#e5e9f0", width=1)
        tick = min_value + value_span * i / 5
        draw.text((x - 22, bottom + 22), f"{tick:.0f}", fill="#6b7280", font=font(20))
    draw.line((zero_x, top, zero_x, bottom), fill="#94a3b8", width=3)
    colors = {
        "Constrained Q-learning policy": "#6E59A5",
        "DF policy search (revenue)": "#2f6f9f",
        "DF policy search (risk-adjusted)": "#e45756",
        "Rolling-28d mean FTO": "#4f7cac",
        "Robust quantile FTO": "#72b7b2",
        "Fuzzy risk-aware FTO": "#8E6BBE",
        "Prev-day FTO": "#9a9a9a",
    }
    row_h = 86
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        y = top + 28 + idx * row_h
        draw.text((90, y + 12), row.method, fill="#263142", font=font(22, True))
        end_x = xcoord(row.mean_revenue)
        x0, x1 = sorted([zero_x, end_x])
        draw.rounded_rectangle((x0, y, x1, y + 48), radius=7, fill=colors.get(row.method, "#4C78A8"))
        label_x = end_x + 12 if row.mean_revenue >= 0 else end_x - 92
        draw.text((label_x, y + 9), f"{row.mean_revenue:.2f}", fill="#263142", font=font(22, True))
        draw.text(
            (left + 410, y + 53),
            f"regret {row.mean_regret:.2f}; CVaR10 {row.cvar_10:.2f}; loss days {int(row.negative_revenue_days)}",
            fill="#5f6b7a",
            font=font(18),
        )
    draw.text((left, bottom + 68), "Mean daily revenue on final 20% test split (EUR/day proxy)", fill="#526070", font=font(22, True))
    draw.text(
        (90, 1010),
        "Boundary: this is a constrained tabular learning baseline, not a deep RL claim; states use only rolling historical profiles available before each test day.",
        fill="#64748b",
        font=font(19),
    )
    image.save(OUT_FIG)


def run():
    daily_rows = []
    days_by_zone = sim.prepare_days()
    for zone, zone_days in days_by_zone.items():
        train, validation, test = policy.split_contexts(zone_days)
        train_for_selection_raw = train + validation
        discretizer = build_discretizer(train_for_selection_raw)
        train = attach_discrete_state_features(train, discretizer)
        validation = attach_discrete_state_features(validation, discretizer)
        test = attach_discrete_state_features(test, discretizer)
        train_for_selection = train + validation
        q_table = train_q_policy(train_for_selection, discretizer)
        for split, contexts in [("train", train), ("validation", validation), ("test", test)]:
            daily_rows.extend(evaluate_q_contexts(contexts, q_table, discretizer, zone, split))

    q_daily = pd.DataFrame(daily_rows)
    if not DECISION_DAILY.exists():
        policy.run()
    decision_daily = pd.read_csv(DECISION_DAILY)
    keep_methods = {
        "Hindsight optimum",
        "Prev-day FTO",
        "Rolling-28d mean FTO",
        "Robust quantile FTO",
        "DF policy search (revenue)",
        "DF policy search (risk-adjusted)",
    }
    decision_daily = decision_daily[decision_daily["method"].isin(keep_methods)].copy()
    daily = pd.concat([decision_daily, q_daily], ignore_index=True)

    # Add the fuzzy soft-computing comparator without changing its implementation.
    import run_opsd_fuzzy_risk_vpp_baseline as fuzzy

    fuzzy_daily = fuzzy.build_daily()
    fuzzy_daily = fuzzy_daily[fuzzy_daily["method"] == "Fuzzy risk-aware FTO"].copy()
    daily = pd.concat([daily, fuzzy_daily], ignore_index=True)

    summary = summarize(daily)
    aggregate = aggregate_test(summary)
    paired = paired_tests(daily)
    daily.to_csv(OUT_DAILY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    aggregate.to_csv(OUT_AGG, index=False)
    paired.to_csv(OUT_PAIRED, index=False)
    plot(aggregate)
    return daily, summary, aggregate, paired


def main():
    _, _, aggregate, paired = run()
    print(OUT_DAILY)
    print(OUT_SUMMARY)
    print(OUT_AGG)
    print(OUT_PAIRED)
    print(OUT_FIG)
    print(aggregate.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
