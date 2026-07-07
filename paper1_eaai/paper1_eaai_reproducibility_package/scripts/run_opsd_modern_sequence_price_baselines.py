from pathlib import Path
from math import erfc, sqrt

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_opsd_graph_temporal_price_ablation import (
    TARGET,
    TIDY,
    RESULTS,
    FIGURES,
    point_metrics,
    split_zone,
)


LOOKBACK = 168
TREND_WINDOW = 24
L2 = 2e-2


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


def fit_standardized_ridge_np(x, y, l2=L2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std
    xs = np.column_stack([np.ones(len(xs)), xs])
    eye = np.eye(xs.shape[1])
    eye[0, 0] = 0.0
    beta = np.linalg.solve(xs.T @ xs + l2 * eye, xs.T @ y)
    return beta, mean, std


def predict_standardized_ridge_np(x, model):
    beta, mean, std = model
    x = np.asarray(x, dtype=float)
    xs = (x - mean) / std
    xs = np.column_stack([np.ones(len(xs)), xs])
    return xs @ beta


def trailing_trend_matrix(seq):
    trend = np.empty_like(seq)
    for i in range(seq.shape[1]):
        start = max(0, i - TREND_WINDOW + 1)
        trend[:, i] = np.mean(seq[:, start : i + 1], axis=1)
    return trend


def make_sequence_frame():
    if not TIDY.exists():
        raise SystemExit(f"Missing processed OPSD file: {TIDY}")
    tidy = pd.read_csv(TIDY, parse_dates=["timestamp_utc"])
    tidy = tidy.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    lag_frame = pd.concat(
        {
            f"price_lag_{lag}": tidy.groupby("zone")[TARGET].shift(lag)
            for lag in range(1, LOOKBACK + 1)
        },
        axis=1,
    )
    tidy = pd.concat([tidy, lag_frame], axis=1)
    tidy["lag_1_24_abs"] = (tidy["price_lag_1"] - tidy["price_lag_24"]).abs()
    return tidy


def paired_sign_test(base_abs, model_abs):
    diff = np.asarray(base_abs, dtype=float) - np.asarray(model_abs, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    n = int(len(diff))
    if n == 0:
        return {
            "paired_n": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_abs_error_delta": np.nan,
            "sign_test_z": np.nan,
            "sign_test_p_approx": np.nan,
        }
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    z = (wins - n / 2) / sqrt(n / 4)
    return {
        "paired_n": n,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n,
        "mean_abs_error_delta": float(np.mean(diff)),
        "sign_test_z": float(z),
        "sign_test_p_approx": float(erfc(abs(z) / sqrt(2))),
    }


def sequence_design(frame):
    lag_cols = [f"price_lag_{lag}" for lag in range(LOOKBACK, 0, -1)]
    seq = frame[lag_cols].to_numpy(float)
    trend = trailing_trend_matrix(seq)
    seasonal = seq - trend
    last = seq[:, -1:]
    nlinear = seq - last
    return {
        "seq": seq,
        "trend": trend,
        "seasonal": seasonal,
        "nlinear": nlinear,
        "last": last.ravel(),
    }


def fit_sequence_models(train):
    design = sequence_design(train)
    y = train[TARGET].to_numpy(float)
    dlinear = fit_standardized_ridge_np(np.column_stack([design["trend"], design["seasonal"]]), y)
    nlinear_resid = fit_standardized_ridge_np(design["nlinear"], y - design["last"])
    return {
        "dlinear": dlinear,
        "nlinear_resid": nlinear_resid,
    }


def predict_sequence_models(frame, models):
    design = sequence_design(frame)
    dlinear = predict_standardized_ridge_np(np.column_stack([design["trend"], design["seasonal"]]), models["dlinear"])
    nlinear = design["last"] + predict_standardized_ridge_np(design["nlinear"], models["nlinear_resid"])
    return dlinear, nlinear


def evaluate_zone(zone, zdf):
    lag_cols = [f"price_lag_{lag}" for lag in range(1, LOOKBACK + 1)]
    clean = zdf[["timestamp_utc", "zone", TARGET, "lag_1_24_abs"] + lag_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train, cal, test = split_zone(clean)
    models = fit_sequence_models(train)
    cal_dlinear, cal_nlinear = predict_sequence_models(cal, models)
    test_dlinear, test_nlinear = predict_sequence_models(test, models)

    cal = cal.copy()
    test = test.copy()
    cal["dlinear_seq_pred"] = cal_dlinear
    cal["nlinear_seq_pred"] = cal_nlinear
    test["dlinear_seq_pred"] = test_dlinear
    test["nlinear_seq_pred"] = test_nlinear

    spike_threshold = float(np.nanquantile(train["lag_1_24_abs"], 0.90))
    test["spike_regime"] = test["lag_1_24_abs"] >= spike_threshold
    model_specs = [
        ("DLinear-style sequence ridge", "dlinear_seq_pred"),
        ("NLinear-style sequence ridge", "nlinear_seq_pred"),
    ]
    rows = []
    paired_rows = []
    for regime, mask in [
        ("all", np.ones(len(test), dtype=bool)),
        ("spike", test["spike_regime"].to_numpy(bool)),
        ("non_spike", ~test["spike_regime"].to_numpy(bool)),
    ]:
        sub = test.loc[mask]
        base_abs = np.abs(sub[TARGET].to_numpy(float) - sub["dlinear_seq_pred"].to_numpy(float))
        for model_name, col in model_specs:
            rows.append(
                {
                    "dataset": "OPSD",
                    "zone": zone,
                    "regime": regime,
                    "model": model_name,
                    "lookback_hours": LOOKBACK,
                    "trend_window_hours": TREND_WINDOW,
                    "spike_threshold": spike_threshold,
                    **point_metrics(sub[TARGET], sub[col]),
                }
            )
        model_abs = np.abs(sub[TARGET].to_numpy(float) - sub["nlinear_seq_pred"].to_numpy(float))
        paired_rows.append(
            {
                "dataset": "OPSD",
                "zone": zone,
                "regime": regime,
                "model": "NLinear-style sequence ridge",
                "baseline": "DLinear-style sequence ridge",
                **paired_sign_test(base_abs, model_abs),
            }
        )

    daily = test[["timestamp_utc", "zone", TARGET, "lag_1_24_abs", "spike_regime"]].copy()
    daily["dlinear_seq_pred"] = test["dlinear_seq_pred"]
    daily["nlinear_seq_pred"] = test["nlinear_seq_pred"]
    return rows, paired_rows, daily


def run():
    tidy = make_sequence_frame()
    rows = []
    paired_rows = []
    daily_parts = []
    for zone, zdf in tidy.groupby("zone", sort=True):
        zone_rows, zone_paired, zone_daily = evaluate_zone(zone, zdf)
        rows.extend(zone_rows)
        paired_rows.extend(zone_paired)
        daily_parts.append(zone_daily)
    summary = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)
    daily = pd.concat(daily_parts, ignore_index=True)
    summary.to_csv(RESULTS / "opsd_modern_sequence_price_baselines_summary.csv", index=False)
    paired.to_csv(RESULTS / "opsd_modern_sequence_price_baselines_paired_tests.csv", index=False)
    daily.to_csv(RESULTS / "opsd_modern_sequence_price_baselines_daily.csv", index=False)
    return summary, paired


def plot_sequence_vs_graphpatch(sequence_summary):
    graphpatch = pd.read_csv(RESULTS / "opsd_deep_graphpatch_price_summary.csv")
    seq_spike = sequence_summary[sequence_summary["regime"] == "spike"].copy()
    gp_spike = graphpatch[
        (graphpatch["regime"] == "spike")
        & graphpatch["model"].isin(["Local ridge", "Calibrated GraphPatch blend"])
    ].copy()
    plot = pd.concat(
        [
            gp_spike[["zone", "model", "rmse"]],
            seq_spike[["zone", "model", "rmse"]],
        ],
        ignore_index=True,
    )
    zones = sorted(plot["zone"].unique())
    models = [
        "Local ridge",
        "DLinear-style sequence ridge",
        "NLinear-style sequence ridge",
        "Calibrated GraphPatch blend",
    ]
    labels = {
        "Local ridge": "Local ridge",
        "DLinear-style sequence ridge": "DLinear-style",
        "NLinear-style sequence ridge": "NLinear-style",
        "Calibrated GraphPatch blend": "GraphPatch blend",
    }
    colors = {
        "Local ridge": "#4C78A8",
        "DLinear-style sequence ridge": "#F58518",
        "NLinear-style sequence ridge": "#B279A2",
        "Calibrated GraphPatch blend": "#54A24B",
    }
    img = Image.new("RGB", (1540, 1120), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "Modern sequence baselines versus GraphPatch on OPSD spikes", fill="#172033", font=font(40, True))
    d.text((88, 112), "One-step DLinear/NLinear-style ridge baselines use a 168-hour lookback window", fill="#5f6b7a", font=font(23))
    left, top, width, height = 145, 190, 1210, 690
    vals = [float(plot[(plot["zone"] == z) & (plot["model"] == m)]["rmse"].iloc[0]) for z in zones for m in models]
    ymax = max(vals) * 1.18
    for i in range(6):
        y = top + height - i * height / 5
        d.line((left, y, left + width, y), fill="#e5e7eb", width=1)
        d.text((left - 76, y - 14), f"{ymax * i / 5:.1f}", fill="#657385", font=font(20))
    group_w = width / len(zones)
    bar_w = min(52, group_w / 7)
    for zi, zone in enumerate(zones):
        gx = left + zi * group_w + 34
        for mi, model in enumerate(models):
            val = float(plot[(plot["zone"] == zone) & (plot["model"] == model)]["rmse"].iloc[0])
            bh = val / ymax * height
            x0 = gx + mi * (bar_w + 8)
            y0 = top + height - bh
            d.rounded_rectangle((x0, y0, x0 + bar_w, top + height), radius=6, fill=colors[model])
            d.text((x0 - 4, y0 - 28), f"{val:.2f}", fill="#1f2937", font=font(18, True))
        d.text((gx + 45, top + height + 36), zone, fill="#1f2937", font=font(23, True), anchor="mm")
    d.line((left, top + height, left + width, top + height), fill="#94a3b8", width=2)
    lx, ly = 190, 945
    for i, model in enumerate(models):
        x = lx + i * 315
        d.rectangle((x, ly, x + 28, ly + 28), fill=colors[model])
        d.text((x + 42, ly - 4), labels[model], fill="#526070", font=font(20))
    d.text((90, 1032), "Source: OPSD public day-ahead prices; final 20% chronological test split; spike threshold fitted on each zone's training split.", fill="#6b7280", font=font(20))
    out = FIGURES / "paper1_fig14_opsd_modern_sequence_vs_graphpatch.png"
    img.save(out)
    return out


def main():
    summary, paired = run()
    print(RESULTS / "opsd_modern_sequence_price_baselines_summary.csv")
    print(RESULTS / "opsd_modern_sequence_price_baselines_paired_tests.csv")
    print(RESULTS / "opsd_modern_sequence_price_baselines_daily.csv")
    print(plot_sequence_vs_graphpatch(summary))


if __name__ == "__main__":
    main()
