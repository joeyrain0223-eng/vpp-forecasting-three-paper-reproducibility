from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import run_opsd_constrained_q_learning_vpp_baseline as qlearn
import run_opsd_decision_focused_policy_search as policy
import run_opsd_fuzzy_risk_vpp_baseline as fuzzy
import run_opsd_genetic_policy_search_vpp_baseline as genetic
import run_opsd_vpp_risk_simulator as sim


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

OUT_DAILY = RESULTS / "opsd_pso_policy_search_vpp_daily.csv"
OUT_SUMMARY = RESULTS / "opsd_pso_policy_search_vpp_summary.csv"
OUT_AGG = RESULTS / "opsd_pso_policy_search_vpp_test_aggregate.csv"
OUT_COEFS = RESULTS / "opsd_pso_policy_search_vpp_coefficients.csv"
OUT_PAIRED = RESULTS / "opsd_pso_policy_search_vpp_paired_tests.csv"
OUT_FIG = FIGURES / "paper3_fig16_opsd_pso_policy_search_vpp.png"

SEED = 20260707
SWARM_SIZE = 22
ITERATIONS = 16
INERTIA = 0.58
COGNITIVE = 1.28
SOCIAL = 1.42
MAX_SELECTION_CONTEXTS = 720
RISK_WEIGHT = 0.50
LOWER = np.asarray([-1.50, -1.50, -10.0, -10.0], dtype=float)
UPPER = np.asarray([1.50, 1.50, 10.0, 10.0], dtype=float)


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


def deterministic_selection_subset(contexts: list[dict]) -> list[dict]:
    if len(contexts) <= MAX_SELECTION_CONTEXTS:
        return contexts
    idx = np.linspace(0, len(contexts) - 1, MAX_SELECTION_CONTEXTS, dtype=int)
    return [contexts[int(i)] for i in idx]


def evaluate_vector(contexts: list[dict], vector: np.ndarray) -> np.ndarray:
    revenues = []
    coef = tuple(float(x) for x in vector)
    for ctx in contexts:
        charge_score, discharge_score = policy.candidate_scores(
            ctx["mean_price"], ctx["std_price"], ctx["mean_net"], coef
        )
        charge, discharge = sim.schedule_two_scores(charge_score, discharge_score)
        revenues.append(genetic.fast_daily_revenue(ctx["price"], charge, discharge, ctx["net_error_mean"]))
    return np.asarray(revenues, dtype=float)


def objective_score(contexts: list[dict], vector: np.ndarray, objective: str) -> tuple[float, float, float]:
    revenues = evaluate_vector(contexts, vector)
    mean_revenue = float(np.mean(revenues))
    cvar_10 = sim.cvar(revenues, 0.10)
    if objective == "revenue":
        score = mean_revenue
    elif objective == "risk_adjusted":
        score = mean_revenue + RISK_WEIGHT * cvar_10
    else:
        raise ValueError(objective)
    return float(score), mean_revenue, float(cvar_10)


def select_pso_policy(contexts: list[dict], objective: str, seed_offset: int) -> dict:
    rng = np.random.default_rng(SEED + seed_offset)
    anchors = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, -8.0, 8.0],
            [1.0, -1.0, 8.0, -8.0],
            [-0.5, 0.5, -4.0, 4.0],
            [0.5, -0.5, 4.0, -4.0],
        ],
        dtype=float,
    )
    random_part = rng.uniform(LOWER, UPPER, size=(SWARM_SIZE - len(anchors), 4))
    position = np.vstack([anchors, random_part])
    velocity = rng.normal(0.0, [0.18, 0.18, 1.0, 1.0], size=position.shape)
    personal_best = position.copy()
    personal_score = np.full(SWARM_SIZE, -np.inf, dtype=float)
    personal_mean = np.zeros(SWARM_SIZE, dtype=float)
    personal_cvar = np.zeros(SWARM_SIZE, dtype=float)
    global_best = None
    history = []

    for iteration in range(ITERATIONS):
        for i in range(SWARM_SIZE):
            score, mean_revenue, cvar_10 = objective_score(contexts, position[i], objective)
            if score > personal_score[i]:
                personal_best[i] = position[i].copy()
                personal_score[i] = score
                personal_mean[i] = mean_revenue
                personal_cvar[i] = cvar_10
            if global_best is None or score > global_best["score"]:
                global_best = {
                    "coef": position[i].copy(),
                    "score": score,
                    "train_mean_revenue": mean_revenue,
                    "train_cvar_10": cvar_10,
                    "iteration": iteration,
                }
        assert global_best is not None
        history.append(float(global_best["score"]))
        r1 = rng.random(position.shape)
        r2 = rng.random(position.shape)
        velocity = (
            INERTIA * velocity
            + COGNITIVE * r1 * (personal_best - position)
            + SOCIAL * r2 * (global_best["coef"] - position)
        )
        position = np.clip(position + velocity, LOWER, UPPER)

    assert global_best is not None
    return {**global_best, "history_best_scores": history}


def evaluate_pso_rows(ctx: dict, zone: str, split: str, selected: dict, label: str) -> dict:
    charge_score, discharge_score = policy.candidate_scores(
        ctx["mean_price"], ctx["std_price"], ctx["mean_net"], tuple(selected["coef"])
    )
    charge, discharge = sim.schedule_two_scores(charge_score, discharge_score)
    return policy.evaluate_schedule(ctx, charge, discharge, ctx["net_error_mean"], label, zone, split)


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
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


def aggregate_test(summary: pd.DataFrame) -> pd.DataFrame:
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


def exact_two_sided_sign_test(wins: int, losses: int) -> float:
    n = int(wins + losses)
    if n == 0:
        return 1.0
    k = min(int(wins), int(losses))
    prob = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2 * prob))


def paired_tests(daily: pd.DataFrame) -> pd.DataFrame:
    test = daily[daily["split"] == "test"].copy()
    key_cols = ["zone", "date"]
    pso = test[test["method"] == "PSO policy search (risk-adjusted)"][key_cols + ["revenue"]].rename(
        columns={"revenue": "pso_revenue"}
    )
    baseline_methods = [
        "PSO policy search (revenue)",
        "Genetic policy search (risk-adjusted)",
        "Genetic policy search (revenue)",
        "DF policy search (revenue)",
        "DF policy search (risk-adjusted)",
        "Robust quantile FTO",
        "Rolling-28d mean FTO",
        "Fuzzy risk-aware FTO",
        "Constrained Q-learning policy",
    ]
    rows = []
    for method in baseline_methods:
        other = test[test["method"] == method][key_cols + ["revenue"]].rename(columns={"revenue": "baseline_revenue"})
        joined = pso.merge(other, on=key_cols, how="inner")
        diff = joined["pso_revenue"].to_numpy(float) - joined["baseline_revenue"].to_numpy(float)
        wins = int(np.sum(diff > 1e-9))
        losses = int(np.sum(diff < -1e-9))
        ties = int(len(diff) - wins - losses)
        rows.append(
            {
                "comparison": f"PSO policy search (risk-adjusted) vs {method}",
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


def plot(aggregate: pd.DataFrame) -> None:
    order = [
        "PSO policy search (risk-adjusted)",
        "PSO policy search (revenue)",
        "Genetic policy search (risk-adjusted)",
        "Genetic policy search (revenue)",
        "DF policy search (revenue)",
        "DF policy search (risk-adjusted)",
        "Robust quantile FTO",
        "Fuzzy risk-aware FTO",
        "Constrained Q-learning policy",
    ]
    plot_df = aggregate[aggregate["method"].isin(order)].copy()
    plot_df["order"] = plot_df["method"].map({m: i for i, m in enumerate(order)})
    plot_df = plot_df.sort_values("order")
    width, height = 2200, 1260
    left, top, right, bottom = 720, 215, 1860, 990
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 54), "Particle-swarm soft-computing policy search for VPP bidding", fill="#172033", font=font(42, True))
    draw.text(
        (92, 112),
        "PSO searches the same auditable coefficient policy as GA under the same chronological OPSD split",
        fill="#5f6b7a",
        font=font(22),
    )
    min_value = min(0.0, float(plot_df["mean_revenue"].min()) * 1.18)
    max_value = max(float(plot_df["mean_revenue"].max()) * 1.18, 1.0)
    span = max_value - min_value

    def xcoord(value):
        return left + (float(value) - min_value) / span * (right - left)

    zero_x = xcoord(0.0)
    for i in range(6):
        x = left + i * (right - left) / 5
        draw.line((x, top, x, bottom), fill="#e5e9f0", width=1)
        tick = min_value + span * i / 5
        draw.text((x - 22, bottom + 22), "0" if abs(tick) < 0.5 else f"{tick:.0f}", fill="#6b7280", font=font(20))
    draw.line((zero_x, top, zero_x, bottom), fill="#94a3b8", width=3)
    colors = {
        "PSO policy search (risk-adjusted)": "#7c3aed",
        "PSO policy search (revenue)": "#a78bfa",
        "Genetic policy search (risk-adjusted)": "#0f766e",
        "Genetic policy search (revenue)": "#14b8a6",
        "DF policy search (revenue)": "#2f6f9f",
        "DF policy search (risk-adjusted)": "#e45756",
        "Robust quantile FTO": "#72b7b2",
        "Fuzzy risk-aware FTO": "#8E6BBE",
        "Constrained Q-learning policy": "#6E59A5",
    }
    row_h = 77
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        y = top + 22 + idx * row_h
        draw.text((90, y + 10), row.method, fill="#263142", font=font(20, True))
        end_x = xcoord(row.mean_revenue)
        x0, x1 = sorted([zero_x, end_x])
        draw.rounded_rectangle((x0, y, x1, y + 43), radius=7, fill=colors.get(row.method, "#4C78A8"))
        label_x = end_x + 12 if row.mean_revenue >= 0 else end_x - 92
        draw.text((label_x, y + 8), f"{row.mean_revenue:.2f}", fill="#263142", font=font(20, True))
        draw.text(
            (left + 465, y + 47),
            f"regret {row.mean_regret:.2f}; CVaR10 {row.cvar_10:.2f}; loss days {int(row.negative_revenue_days)}",
            fill="#5f6b7a",
            font=font(16),
        )
    draw.text((left, bottom + 66), "Mean daily revenue on final 20% test split (EUR/day proxy)", fill="#526070", font=font(22, True))
    draw.text(
        (90, 1160),
        "Boundary: PSO is a reviewer-facing swarm baseline for the same transparent policy class, not evidence of a production-grade market engine.",
        fill="#64748b",
        font=font(19),
    )
    image.save(OUT_FIG)


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_rows = []
    coef_rows = []
    days_by_zone = sim.prepare_days()
    for zone_idx, (zone, zone_days) in enumerate(days_by_zone.items()):
        train, validation, test = policy.split_contexts(zone_days)
        full_selection_contexts = train + validation
        selection_contexts = deterministic_selection_subset(full_selection_contexts)
        selected = {
            "PSO policy search (revenue)": select_pso_policy(selection_contexts, "revenue", zone_idx * 10 + 1),
            "PSO policy search (risk-adjusted)": select_pso_policy(selection_contexts, "risk_adjusted", zone_idx * 10 + 2),
        }
        for label, record in selected.items():
            c_std, d_std, c_net, d_net = [float(x) for x in record["coef"]]
            coef_rows.append(
                {
                    "zone": zone,
                    "method": label,
                    "charge_std_coef": c_std,
                    "discharge_std_coef": d_std,
                    "charge_net_coef": c_net,
                    "discharge_net_coef": d_net,
                    "selection_score": record["score"],
                    "selection_mean_revenue": record["train_mean_revenue"],
                    "selection_cvar_10": record["train_cvar_10"],
                    "best_iteration": record["iteration"],
                    "swarm_size": SWARM_SIZE,
                    "iterations": ITERATIONS,
                    "train_validation_days": len(full_selection_contexts),
                    "selection_subset_days": len(selection_contexts),
                    "test_days": len(test),
                }
            )
        for split, contexts in [("train", train), ("validation", validation), ("test", test)]:
            for ctx in contexts:
                for label, record in selected.items():
                    daily_rows.append(evaluate_pso_rows(ctx, zone, split, record, label))

    pso_daily = pd.DataFrame(daily_rows)
    decision_daily_path = RESULTS / "opsd_decision_focused_policy_daily.csv"
    if not decision_daily_path.exists():
        policy.run()
    decision_daily = pd.read_csv(decision_daily_path)
    decision_keep = {
        "Hindsight optimum",
        "Prev-day FTO",
        "Rolling-28d mean FTO",
        "Robust quantile FTO",
        "DF policy search (revenue)",
        "DF policy search (risk-adjusted)",
    }
    decision_daily = decision_daily[decision_daily["method"].isin(decision_keep)].copy()
    fuzzy_daily = fuzzy.build_daily()
    fuzzy_daily = fuzzy_daily[fuzzy_daily["method"] == "Fuzzy risk-aware FTO"].copy()
    if not qlearn.OUT_DAILY.exists():
        qlearn.run()
    q_daily = pd.read_csv(qlearn.OUT_DAILY)
    q_daily = q_daily[q_daily["method"] == "Constrained Q-learning policy"].copy()
    if not genetic.OUT_DAILY.exists():
        genetic.run()
    genetic_daily = pd.read_csv(genetic.OUT_DAILY)
    genetic_daily = genetic_daily[
        genetic_daily["method"].isin({"Genetic policy search (revenue)", "Genetic policy search (risk-adjusted)"})
    ].copy()
    daily = pd.concat([decision_daily, fuzzy_daily, q_daily, genetic_daily, pso_daily], ignore_index=True)
    summary = summarize(daily)
    aggregate = aggregate_test(summary)
    coefs = pd.DataFrame(coef_rows)
    paired = paired_tests(daily)
    daily.to_csv(OUT_DAILY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    aggregate.to_csv(OUT_AGG, index=False)
    coefs.to_csv(OUT_COEFS, index=False)
    paired.to_csv(OUT_PAIRED, index=False)
    plot(aggregate)
    return daily, summary, aggregate, coefs, paired


def main() -> None:
    _, _, aggregate, coefs, paired = run()
    print(OUT_DAILY)
    print(OUT_SUMMARY)
    print(OUT_AGG)
    print(OUT_COEFS)
    print(OUT_PAIRED)
    print(OUT_FIG)
    print(aggregate.to_string(index=False))
    print(coefs.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
