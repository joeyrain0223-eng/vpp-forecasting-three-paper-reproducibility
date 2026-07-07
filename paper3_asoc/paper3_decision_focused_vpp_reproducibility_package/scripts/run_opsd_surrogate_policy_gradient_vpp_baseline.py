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
import run_opsd_pso_policy_search_vpp_baseline as pso
import run_opsd_vpp_risk_simulator as sim


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

OUT_DAILY = RESULTS / "opsd_surrogate_policy_gradient_vpp_daily.csv"
OUT_SUMMARY = RESULTS / "opsd_surrogate_policy_gradient_vpp_summary.csv"
OUT_AGG = RESULTS / "opsd_surrogate_policy_gradient_vpp_test_aggregate.csv"
OUT_COEFS = RESULTS / "opsd_surrogate_policy_gradient_vpp_coefficients.csv"
OUT_TRACE = RESULTS / "opsd_surrogate_policy_gradient_vpp_training_trace.csv"
OUT_PAIRED = RESULTS / "opsd_surrogate_policy_gradient_vpp_paired_tests.csv"
OUT_FIG = FIGURES / "paper3_fig17_opsd_surrogate_policy_gradient_vpp.png"

FEATURE_DIM = 6
THETA_DIM = FEATURE_DIM * 2
ACTIVE = sim.ACTIVE_HOURS
TEMPERATURE = 0.72
RISK_WEIGHT = 0.45
OVERLAP_PENALTY = 7.50
MAX_SELECTION_CONTEXTS = 192
ITERATIONS = 36
FD_EPS = 0.035
LR0 = 0.18
ADAM_BETA1 = 0.86
ADAM_BETA2 = 0.97
ADAM_EPS = 1e-7


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


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    shifted = x - float(np.max(x))
    exp = np.exp(shifted)
    return exp / max(float(exp.sum()), 1e-12)


def normalize_hourly(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(values.std())
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def features_for(ctx: dict) -> np.ndarray:
    if "surrogate_features" in ctx:
        return np.asarray(ctx["surrogate_features"], dtype=float)
    hours = np.arange(24, dtype=float)
    price_z = normalize_hourly(ctx["mean_price"])
    std_z = normalize_hourly(ctx["std_price"])
    net_z = policy.net_shape(ctx["mean_net"])
    sin_h = np.sin(2.0 * np.pi * hours / 24.0)
    cos_h = np.cos(2.0 * np.pi * hours / 24.0)
    bias = np.ones(24, dtype=float)
    return np.column_stack([price_z, std_z, net_z, sin_h, cos_h, bias])


def attach_cached_features(contexts: list[dict]) -> list[dict]:
    cached = []
    for ctx in contexts:
        copied = dict(ctx)
        copied["surrogate_features"] = features_for(ctx)
        cached.append(copied)
    return cached


def split_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    return theta[:FEATURE_DIM], theta[FEATURE_DIM:]


def surrogate_revenue(ctx: dict, theta: np.ndarray) -> float:
    wc, wd = split_theta(theta)
    x = features_for(ctx)
    charge_logits = x @ wc
    discharge_logits = x @ wd
    charge_weight = ACTIVE * stable_softmax(-charge_logits / TEMPERATURE)
    discharge_weight = ACTIVE * stable_softmax(discharge_logits / TEMPERATURE)
    price = np.asarray(ctx["price"], dtype=float)
    net_error = np.abs(np.asarray(ctx["net_error_mean"], dtype=float))
    battery_power = sim.BATTERY_CAPACITY_MWH / ACTIVE
    flex_power = sim.FLEX_LOAD_MWH / ACTIVE
    battery_revenue = battery_power * (
        sim.EFFICIENCY * float(np.dot(discharge_weight, price))
        - float(np.dot(charge_weight, price)) / sim.EFFICIENCY
    )
    flex_value = flex_power * (
        float(np.dot(discharge_weight, price)) - float(np.dot(charge_weight, price))
    )
    active_weight = charge_weight + discharge_weight
    imbalance_penalty = sim.IMBALANCE_PENALTY_RATE * (battery_power + flex_power) * float(
        np.dot(active_weight, net_error) / (2.0 * ACTIVE)
    )
    overlap_penalty = OVERLAP_PENALTY * float(np.dot(charge_weight, discharge_weight) / (ACTIVE * ACTIVE))
    return float(battery_revenue + flex_value - imbalance_penalty - overlap_penalty)


def objective(contexts: list[dict], theta: np.ndarray, objective_name: str) -> tuple[float, float, float]:
    revenues = np.asarray([surrogate_revenue(ctx, theta) for ctx in contexts], dtype=float)
    mean_revenue = float(np.mean(revenues))
    cvar_10 = sim.cvar(revenues, 0.10)
    if objective_name == "revenue":
        score = mean_revenue
    elif objective_name == "risk_adjusted":
        score = mean_revenue + RISK_WEIGHT * cvar_10
    else:
        raise ValueError(objective_name)
    return float(score), mean_revenue, float(cvar_10)


def finite_difference_gradient(contexts: list[dict], theta: np.ndarray, objective_name: str) -> np.ndarray:
    grad = np.zeros_like(theta, dtype=float)
    for idx in range(len(theta)):
        step = np.zeros_like(theta, dtype=float)
        step[idx] = FD_EPS
        plus = objective(contexts, theta + step, objective_name)[0]
        minus = objective(contexts, theta - step, objective_name)[0]
        grad[idx] = (plus - minus) / (2.0 * FD_EPS)
    return grad


def deterministic_selection_subset(contexts: list[dict]) -> list[dict]:
    if len(contexts) <= MAX_SELECTION_CONTEXTS:
        return attach_cached_features(contexts)
    idx = np.linspace(0, len(contexts) - 1, MAX_SELECTION_CONTEXTS, dtype=int)
    return attach_cached_features([contexts[int(i)] for i in idx])


def initial_theta(objective_name: str) -> np.ndarray:
    # Start from a transparent price-spread prior and let the surrogate update
    # volatility, net-shape, and intraday terms on chronological public data.
    if objective_name == "risk_adjusted":
        wc = np.asarray([1.0, 0.42, -0.30, 0.0, 0.0, 0.0], dtype=float)
        wd = np.asarray([1.0, -0.42, 0.30, 0.0, 0.0, 0.0], dtype=float)
    else:
        wc = np.asarray([1.0, 0.18, -0.18, 0.0, 0.0, 0.0], dtype=float)
        wd = np.asarray([1.0, -0.18, 0.18, 0.0, 0.0, 0.0], dtype=float)
    return np.concatenate([wc, wd])


def train_surrogate_policy(contexts: list[dict], objective_name: str) -> tuple[np.ndarray, dict, list[dict]]:
    theta = initial_theta(objective_name)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    trace = []
    best_theta = theta.copy()
    best = {"score": -np.inf, "train_mean_revenue": np.nan, "train_cvar_10": np.nan, "iteration": 0}
    for iteration in range(1, ITERATIONS + 1):
        score, mean_revenue, cvar_10 = objective(contexts, theta, objective_name)
        if score > best["score"]:
            best = {
                "score": float(score),
                "train_mean_revenue": float(mean_revenue),
                "train_cvar_10": float(cvar_10),
                "iteration": iteration,
            }
            best_theta = theta.copy()
        trace.append(
            {
                "iteration": iteration,
                "objective": objective_name,
                "surrogate_score": float(score),
                "surrogate_mean_revenue": float(mean_revenue),
                "surrogate_cvar_10": float(cvar_10),
                "learning_rate": LR0 / np.sqrt(iteration),
            }
        )
        grad = finite_difference_gradient(contexts, theta, objective_name)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 50.0:
            grad = grad * (50.0 / grad_norm)
        m = ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * grad
        v = ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * (grad * grad)
        m_hat = m / (1.0 - ADAM_BETA1**iteration)
        v_hat = v / (1.0 - ADAM_BETA2**iteration)
        lr = LR0 / np.sqrt(iteration)
        theta = theta + lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
        theta = np.clip(theta, -5.0, 5.0)
    return best_theta, best, trace


def discrete_schedule(ctx: dict, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wc, wd = split_theta(theta)
    x = features_for(ctx)
    charge_score = x @ wc
    discharge_score = x @ wd
    return sim.schedule_two_scores(charge_score, discharge_score)


def evaluate_surrogate_row(ctx: dict, zone: str, split: str, theta: np.ndarray, label: str) -> dict:
    charge, discharge = discrete_schedule(ctx, theta)
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
    surrogate = test[test["method"] == "Surrogate policy-gradient (risk-adjusted)"][key_cols + ["revenue"]].rename(
        columns={"revenue": "surrogate_revenue"}
    )
    baseline_methods = [
        "Surrogate policy-gradient (revenue)",
        "Genetic policy search (risk-adjusted)",
        "PSO policy search (risk-adjusted)",
        "DF policy search (revenue)",
        "DF policy search (risk-adjusted)",
        "Robust quantile FTO",
        "Fuzzy risk-aware FTO",
        "Constrained Q-learning policy",
    ]
    rows = []
    for method in baseline_methods:
        other = test[test["method"] == method][key_cols + ["revenue"]].rename(columns={"revenue": "baseline_revenue"})
        joined = surrogate.merge(other, on=key_cols, how="inner")
        diff = joined["surrogate_revenue"].to_numpy(float) - joined["baseline_revenue"].to_numpy(float)
        wins = int(np.sum(diff > 1e-9))
        losses = int(np.sum(diff < -1e-9))
        ties = int(len(diff) - wins - losses)
        rows.append(
            {
                "comparison": f"Surrogate policy-gradient (risk-adjusted) vs {method}",
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


def ensure_upstream_daily() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not genetic.OUT_DAILY.exists():
        genetic.run()
    if not pso.OUT_DAILY.exists():
        pso.run()
    if not qlearn.OUT_DAILY.exists():
        qlearn.run()
    genetic_daily = pd.read_csv(genetic.OUT_DAILY)
    pso_daily = pd.read_csv(pso.OUT_DAILY)
    q_daily = pd.read_csv(qlearn.OUT_DAILY)
    fuzzy_daily = fuzzy.build_daily()
    return genetic_daily, pso_daily, q_daily, fuzzy_daily


def plot(aggregate: pd.DataFrame) -> None:
    order = [
        "Genetic policy search (risk-adjusted)",
        "PSO policy search (risk-adjusted)",
        "Surrogate policy-gradient (risk-adjusted)",
        "Surrogate policy-gradient (revenue)",
        "DF policy search (revenue)",
        "DF policy search (risk-adjusted)",
        "Robust quantile FTO",
        "Fuzzy risk-aware FTO",
        "Constrained Q-learning policy",
    ]
    plot_df = aggregate[aggregate["method"].isin(order)].copy()
    plot_df["order"] = plot_df["method"].map({m: i for i, m in enumerate(order)})
    plot_df = plot_df.sort_values("order")

    width, height = 2200, 1280
    left, top, right, bottom = 760, 220, 1880, 990
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 56), "Surrogate policy-gradient baseline for public VPP bidding", fill="#172033", font=font(43, True))
    draw.text(
        (92, 115),
        "A differentiable soft-schedule surrogate is trained on chronological OPSD train/validation days and frozen before final test evaluation",
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
        draw.text((x - 24, bottom + 25), "0" if abs(tick) < 0.5 else f"{tick:.0f}", fill="#6b7280", font=font(20))
    draw.line((zero_x, top, zero_x, bottom), fill="#94a3b8", width=3)

    colors = {
        "Genetic policy search (risk-adjusted)": "#0f766e",
        "PSO policy search (risk-adjusted)": "#2563eb",
        "Surrogate policy-gradient (risk-adjusted)": "#c2410c",
        "Surrogate policy-gradient (revenue)": "#f97316",
        "DF policy search (revenue)": "#2f6f9f",
        "DF policy search (risk-adjusted)": "#e45756",
        "Robust quantile FTO": "#72b7b2",
        "Fuzzy risk-aware FTO": "#8E6BBE",
        "Constrained Q-learning policy": "#6E59A5",
    }
    row_h = 75
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        y = top + 24 + idx * row_h
        draw.text((90, y + 8), row.method, fill="#263142", font=font(20, True))
        end_x = xcoord(row.mean_revenue)
        x0, x1 = sorted([zero_x, end_x])
        draw.rounded_rectangle((x0, y, x1, y + 44), radius=7, fill=colors.get(row.method, "#4C78A8"))
        label_x = end_x + 12 if row.mean_revenue >= 0 else end_x - 94
        draw.text((label_x, y + 7), f"{row.mean_revenue:.2f}", fill="#263142", font=font(20, True))
        draw.text(
            (left + 455, y + 48),
            f"regret {row.mean_regret:.2f}; CVaR10 {row.cvar_10:.2f}; loss days {int(row.negative_revenue_days)}",
            fill="#5f6b7a",
            font=font(16),
        )
    draw.text((left, bottom + 68), "Mean daily revenue on final 20% test split (EUR/day proxy)", fill="#526070", font=font(22, True))
    draw.text(
        (90, 1180),
        "Boundary: this is a lightweight differentiable surrogate-policy baseline, not an end-to-end neural-RL market engine or a replacement for external market validation.",
        fill="#64748b",
        font=font(19),
    )
    image.save(OUT_FIG)


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_rows = []
    coef_rows = []
    trace_rows = []
    days_by_zone = sim.prepare_days()
    for zone_idx, (zone, zone_days) in enumerate(days_by_zone.items()):
        train, validation, test = policy.split_contexts(zone_days)
        full_selection_contexts = train + validation
        selection_contexts = deterministic_selection_subset(full_selection_contexts)
        selected = {}
        for objective_name, label in [
            ("revenue", "Surrogate policy-gradient (revenue)"),
            ("risk_adjusted", "Surrogate policy-gradient (risk-adjusted)"),
        ]:
            theta, info, trace = train_surrogate_policy(selection_contexts, objective_name)
            selected[label] = theta
            for row in trace:
                trace_rows.append({"zone": zone, **row})
            wc, wd = split_theta(theta)
            coef_rows.append(
                {
                    "zone": zone,
                    "method": label,
                    "objective": objective_name,
                    "selection_score": info["score"],
                    "selection_mean_revenue": info["train_mean_revenue"],
                    "selection_cvar_10": info["train_cvar_10"],
                    "best_iteration": info["iteration"],
                    "train_validation_days": len(full_selection_contexts),
                    "selection_subset_days": len(selection_contexts),
                    "test_days": len(test),
                    **{f"charge_w_{idx}": float(value) for idx, value in enumerate(wc)},
                    **{f"discharge_w_{idx}": float(value) for idx, value in enumerate(wd)},
                }
            )
        for split, contexts in [("train", train), ("validation", validation), ("test", test)]:
            for ctx in contexts:
                for label, theta in selected.items():
                    daily_rows.append(evaluate_surrogate_row(ctx, zone, split, theta, label))

    surrogate_daily = pd.DataFrame(daily_rows)
    genetic_daily, pso_daily, q_daily, fuzzy_daily = ensure_upstream_daily()
    keep_methods = {
        "Hindsight optimum",
        "Prev-day FTO",
        "Rolling-28d mean FTO",
        "Robust quantile FTO",
        "DF policy search (revenue)",
        "DF policy search (risk-adjusted)",
        "Fuzzy risk-aware FTO",
        "Constrained Q-learning policy",
        "Genetic policy search (risk-adjusted)",
        "PSO policy search (risk-adjusted)",
    }
    upstream = pd.concat([genetic_daily, pso_daily, q_daily, fuzzy_daily], ignore_index=True)
    upstream = upstream[upstream["method"].isin(keep_methods)].copy()
    daily = pd.concat([upstream, surrogate_daily], ignore_index=True)
    daily = daily.drop_duplicates(subset=["split", "zone", "date", "method"], keep="last")
    summary = summarize(daily)
    aggregate = aggregate_test(summary)
    coefs = pd.DataFrame(coef_rows)
    trace = pd.DataFrame(trace_rows)
    paired = paired_tests(daily)
    daily.to_csv(OUT_DAILY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    aggregate.to_csv(OUT_AGG, index=False)
    coefs.to_csv(OUT_COEFS, index=False)
    trace.to_csv(OUT_TRACE, index=False)
    paired.to_csv(OUT_PAIRED, index=False)
    plot(aggregate)
    return daily, summary, aggregate, coefs, paired


def main() -> None:
    _, _, aggregate, coefs, paired = run()
    print(OUT_DAILY)
    print(OUT_SUMMARY)
    print(OUT_AGG)
    print(OUT_COEFS)
    print(OUT_TRACE)
    print(OUT_PAIRED)
    print(OUT_FIG)
    print(aggregate.to_string(index=False))
    print(coefs.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
