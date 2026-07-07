from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SOURCE = RESULTS / "uci_ssl_cold_start_results.csv"
TESTS_OUT = RESULTS / "uci_ssl_client_level_stat_tests.csv"
DELTAS_OUT = RESULTS / "uci_ssl_client_level_rmse_deltas.csv"
FIGURE_OUT = FIGURES / "paper2_fig8_uci_client_level_stat_tests.png"


COMPARISONS = [
    {
        "comparison": "zero-label SSL source head vs 28d target ridge",
        "candidate": "SSL-MR-lag-source-head",
        "baseline": "Target-linear-28d",
        "interpretation": "cold-start source representation versus strongest target-only transparent baseline",
    },
    {
        "comparison": "28d SSL adapter vs 28d target ridge",
        "candidate": "SSL-MR-lag+adapter-28d",
        "baseline": "Target-linear-28d",
        "interpretation": "few-shot representation adaptation versus same-label-budget target ridge",
    },
    {
        "comparison": "7d SSL adapter vs 7d target ridge",
        "candidate": "SSL-MR-lag+adapter-7d",
        "baseline": "Target-linear-7d",
        "interpretation": "one-week adapter versus one-week target ridge",
    },
    {
        "comparison": "1d SSL adapter vs 1d target ridge",
        "candidate": "SSL-MR-lag+adapter-1d",
        "baseline": "Target-linear-1d",
        "interpretation": "one-day adapter versus one-day target ridge",
    },
    {
        "comparison": "zero-label SSL source head vs seasonal-168h",
        "candidate": "SSL-MR-lag-source-head",
        "baseline": "Seasonal-168h",
        "interpretation": "source representation versus weekly seasonal rule",
    },
    {
        "comparison": "zero-label SSL source head vs seasonal-24h",
        "candidate": "SSL-MR-lag-source-head",
        "baseline": "Seasonal-24h",
        "interpretation": "source representation versus daily seasonal rule",
    },
]


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


def exact_two_sided_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def comparison_stats(frame: pd.DataFrame, candidate: str, baseline: str, comparison: str, interpretation: str) -> tuple[dict, pd.DataFrame]:
    cand = frame[frame["model"] == candidate][["target_client", "mae", "rmse", "smape"]].rename(
        columns={"mae": "candidate_mae", "rmse": "candidate_rmse", "smape": "candidate_smape"}
    )
    base = frame[frame["model"] == baseline][["target_client", "mae", "rmse", "smape"]].rename(
        columns={"mae": "baseline_mae", "rmse": "baseline_rmse", "smape": "baseline_smape"}
    )
    merged = base.merge(cand, on="target_client", how="inner")
    if merged.empty:
        raise ValueError(f"No matched clients for {candidate} vs {baseline}")
    merged["comparison"] = comparison
    merged["baseline_model"] = baseline
    merged["candidate_model"] = candidate
    merged["rmse_delta"] = merged["baseline_rmse"] - merged["candidate_rmse"]
    merged["mae_delta"] = merged["baseline_mae"] - merged["candidate_mae"]
    merged["smape_delta"] = merged["baseline_smape"] - merged["candidate_smape"]
    merged["rmse_improvement_pct"] = merged["rmse_delta"] / merged["baseline_rmse"] * 100.0
    merged["mae_improvement_pct"] = merged["mae_delta"] / merged["baseline_mae"] * 100.0
    wins = int((merged["rmse_delta"] > 1e-12).sum())
    losses = int((merged["rmse_delta"] < -1e-12).sum())
    ties = int(len(merged) - wins - losses)
    rmse_delta = merged["rmse_delta"].to_numpy(float)
    std = float(np.std(rmse_delta, ddof=1)) if len(rmse_delta) > 1 else float("nan")
    row = {
        "comparison": comparison,
        "baseline_model": baseline,
        "candidate_model": candidate,
        "interpretation": interpretation,
        "target_clients": int(len(merged)),
        "mean_baseline_rmse": float(merged["baseline_rmse"].mean()),
        "mean_candidate_rmse": float(merged["candidate_rmse"].mean()),
        "mean_rmse_delta": float(merged["rmse_delta"].mean()),
        "median_rmse_delta": float(merged["rmse_delta"].median()),
        "mean_rmse_improvement_pct": float(merged["rmse_improvement_pct"].mean()),
        "median_rmse_improvement_pct": float(merged["rmse_improvement_pct"].median()),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_excluding_ties": float(wins / (wins + losses)) if wins + losses else float("nan"),
        "sign_test_p_two_sided": exact_two_sided_sign_p(wins, losses),
        "paired_effect_mean_over_sd": float(np.mean(rmse_delta) / std) if std and np.isfinite(std) and std > 0 else float("nan"),
    }
    cols = [
        "comparison",
        "target_client",
        "baseline_model",
        "candidate_model",
        "baseline_rmse",
        "candidate_rmse",
        "rmse_delta",
        "rmse_improvement_pct",
        "baseline_mae",
        "candidate_mae",
        "mae_delta",
        "mae_improvement_pct",
    ]
    return row, merged[cols].sort_values(["comparison", "target_client"])


def plot_deltas(delta: pd.DataFrame, tests: pd.DataFrame) -> None:
    selected = [
        "zero-label SSL source head vs 28d target ridge",
        "28d SSL adapter vs 28d target ridge",
        "7d SSL adapter vs 7d target ridge",
    ]
    plot = delta[delta["comparison"].isin(selected)].copy()
    clients = sorted(plot["target_client"].unique().tolist())
    width, height = 2100, 1260
    left, top, right, bottom = 280, 190, 1860, 930
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "Client-level paired RMSE evidence on UCI targets", fill="#172033", font=font(44, True))
    d.text((94, 112), "Positive values mean the SSL representation has lower RMSE than the paired baseline", fill="#5f6b7a", font=font(24))

    max_abs = float(max(abs(plot["rmse_improvement_pct"].min()), abs(plot["rmse_improvement_pct"].max())) * 1.15)
    max_abs = max(max_abs, 5)

    def y_pos(value):
        return bottom - (value + max_abs) / (2 * max_abs) * (bottom - top)

    zero_y = y_pos(0)
    for i in range(7):
        value = -max_abs + 2 * max_abs * i / 6
        y = y_pos(value)
        d.line((left, y, right, y), fill="#E5E7EB", width=1)
        d.text((left - 125, y - 12), f"{value:.0f}%", fill="#526070", font=font(20))
    d.line((left, zero_y, right, zero_y), fill="#111827", width=2)

    colors = {
        selected[0]: "#4C78A8",
        selected[1]: "#54A24B",
        selected[2]: "#F58518",
    }
    group_w = (right - left) / len(clients)
    bar_w = min(34, group_w / 5)
    for idx, client in enumerate(clients):
        cx = left + group_w * (idx + 0.5)
        d.text((cx - d.textlength(client, font=font(18, True)) / 2, bottom + 22), client, fill="#1f2937", font=font(18, True))
        for j, comp in enumerate(selected):
            row = plot[(plot["target_client"] == client) & (plot["comparison"] == comp)]
            if row.empty:
                continue
            val = float(row["rmse_improvement_pct"].iloc[0])
            x0 = cx + (j - 1) * (bar_w + 6) - bar_w / 2
            x1 = x0 + bar_w
            y = y_pos(val)
            if val >= 0:
                d.rectangle((x0, y, x1, zero_y), fill=colors[comp])
            else:
                d.rectangle((x0, zero_y, x1, y), fill="#D95F5F")

    legend_y = 1000
    for i, comp in enumerate(selected):
        x = 210 + i * 610
        d.rectangle((x, legend_y, x + 30, legend_y + 30), fill=colors[comp])
        short = comp.replace("zero-label SSL source head", "source head").replace("target ridge", "ridge")
        d.text((x + 42, legend_y - 2), short, fill="#374151", font=font(20))

    tests_sel = tests[tests["comparison"].isin(selected)].copy()
    summary = "; ".join(
        f"{row.comparison.split(' vs ')[0]}: {int(row.wins)}/{int(row.target_clients)} wins, p={row.sign_test_p_two_sided:.3f}"
        for row in tests_sel.itertuples(index=False)
    )
    d.text((90, 1080), summary, fill="#4b5563", font=font(20))
    d.text((90, 1130), "Source: UCI Electricity Load Diagrams 2011-2014; paired across ten held-out target clients.", fill="#6b7280", font=font(20))
    img.save(FIGURE_OUT)


def main() -> None:
    frame = pd.read_csv(SOURCE)
    rows = []
    deltas = []
    for spec in COMPARISONS:
        row, delta = comparison_stats(frame, spec["candidate"], spec["baseline"], spec["comparison"], spec["interpretation"])
        rows.append(row)
        deltas.append(delta)
    tests = pd.DataFrame(rows)
    delta = pd.concat(deltas, ignore_index=True)
    tests.to_csv(TESTS_OUT, index=False)
    delta.to_csv(DELTAS_OUT, index=False)
    plot_deltas(delta, tests)
    print(TESTS_OUT)
    print(DELTAS_OUT)
    print(FIGURE_OUT)


if __name__ == "__main__":
    main()
