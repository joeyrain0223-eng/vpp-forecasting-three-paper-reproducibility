from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TIDY = PROCESSED / "opsd_hourly_price_load_renewables_tidy.csv"
TARGET = "price_eur_mwh"
ALPHA = 0.10

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


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


def smape(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    den = np.abs(y) + np.abs(pred)
    out = np.zeros_like(den, dtype=float)
    mask = den != 0
    out[mask] = 2 * np.abs(y[mask] - pred[mask]) / den[mask]
    return float(np.mean(out) * 100)


def point_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = y - pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "smape": smape(y, pred),
        "n": int(len(y)),
    }


def pinball_loss(y, q_pred, tau):
    y = np.asarray(y, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    diff = y - q_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def interval_metrics(y, pred, lower, upper, alpha=ALPHA):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    point = point_metrics(y, pred)
    width = upper - lower
    y_range = max(float(np.nanmax(y) - np.nanmin(y)), 1e-6)
    under = y < lower
    over = y > upper
    interval_score = width + (2 / alpha) * (lower - y) * under + (2 / alpha) * (y - upper) * over
    return {
        **point,
        "picp": float(np.mean((y >= lower) & (y <= upper))),
        "pinaw": float(np.mean(width) / y_range),
        "mean_width": float(np.mean(width)),
        "pinball_05": pinball_loss(y, lower, alpha / 2),
        "pinball_50": pinball_loss(y, pred, 0.5),
        "pinball_95": pinball_loss(y, upper, 1 - alpha / 2),
        "mean_interval_score": float(np.mean(interval_score)),
    }


def fit_standardized_ridge(frame, feature_cols, target_col=TARGET, l2=1e-4):
    x = frame[feature_cols].to_numpy(float)
    y = frame[target_col].to_numpy(float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-8] = 1.0
    x = (x - mean) / std
    x = np.column_stack([np.ones(len(x)), x])
    eye = np.eye(x.shape[1])
    eye[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + l2 * eye, x.T @ y)
    return beta, mean, std


def predict_standardized_ridge(frame, feature_cols, model):
    beta, mean, std = model
    x = frame[feature_cols].to_numpy(float)
    x = (x - mean) / std
    x = np.column_stack([np.ones(len(x)), x])
    return x @ beta


def conformal_quantile(abs_resid, alpha=ALPHA):
    abs_resid = np.asarray(abs_resid, dtype=float)
    abs_resid = abs_resid[np.isfinite(abs_resid)]
    if len(abs_resid) == 0:
        return np.nan
    q_level = min(1.0, np.ceil((len(abs_resid) + 1) * (1 - alpha)) / len(abs_resid))
    return float(np.quantile(abs_resid, q_level, method="higher"))


def split_zone(zdf):
    zdf = zdf.sort_values("timestamp_utc").reset_index(drop=True)
    n = len(zdf)
    train_end = int(n * 0.60)
    cal_end = int(n * 0.80)
    return zdf.iloc[:train_end].copy(), zdf.iloc[train_end:cal_end].copy(), zdf.iloc[cal_end:].copy()


def prepare_frame():
    if not TIDY.exists():
        raise SystemExit(f"Missing processed OPSD file: {TIDY}")
    tidy = pd.read_csv(TIDY, parse_dates=["timestamp_utc"])
    tidy = tidy.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    for lag in [1, 24, 168]:
        tidy[f"price_lag_{lag}"] = tidy.groupby("zone")[TARGET].shift(lag)
    tidy["lag_1_24_abs"] = (tidy["price_lag_1"] - tidy["price_lag_24"]).abs()

    pivot = tidy.pivot_table(index="timestamp_utc", columns="zone", values=TARGET, aggfunc="mean").sort_index()
    for lag in [1, 24, 168]:
        shifted = pivot.shift(lag)
        mean_rows = []
        std_rows = []
        for zone in pivot.columns:
            others = [col for col in pivot.columns if col != zone]
            tmp = shifted[others]
            mean_rows.append(
                tmp.mean(axis=1).rename(f"graph_mean_lag_{lag}_{zone}")
            )
            std_rows.append(
                tmp.std(axis=1).fillna(0).rename(f"graph_std_lag_{lag}_{zone}")
            )
        graph = pd.concat(mean_rows + std_rows, axis=1)
        tidy = tidy.merge(graph.reset_index(), on="timestamp_utc", how="left")

    for lag in [1, 24, 168]:
        tidy[f"graph_mean_lag_{lag}"] = np.nan
        tidy[f"graph_std_lag_{lag}"] = np.nan
        for zone in sorted(tidy["zone"].dropna().unique()):
            mask = tidy["zone"] == zone
            tidy.loc[mask, f"graph_mean_lag_{lag}"] = tidy.loc[mask, f"graph_mean_lag_{lag}_{zone}"]
            tidy.loc[mask, f"graph_std_lag_{lag}"] = tidy.loc[mask, f"graph_std_lag_{lag}_{zone}"]
        tidy = tidy.drop(columns=[c for c in tidy.columns if c.startswith(f"graph_mean_lag_{lag}_") or c.startswith(f"graph_std_lag_{lag}_")])
    return tidy


def evaluate_zone(zone, zdf):
    base_features = [
        "price_lag_1",
        "price_lag_24",
        "price_lag_168",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month",
        "load_mw",
        "solar_mw",
        "wind_mw",
    ]
    graph_features = [
        "base_pred",
        "price_lag_1",
        "price_lag_24",
        "price_lag_168",
        "lag_1_24_abs",
        "graph_mean_lag_1",
        "graph_mean_lag_24",
        "graph_mean_lag_168",
        "graph_std_lag_1",
        "graph_std_lag_24",
        "graph_std_lag_168",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month",
        "load_mw",
        "solar_mw",
        "wind_mw",
    ]
    source_graph_features = [col for col in graph_features if col != "base_pred"]
    cols = list(dict.fromkeys(["timestamp_utc", "zone", TARGET, "hour", "lag_1_24_abs"] + sorted(set(base_features + source_graph_features))))
    clean = zdf[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train, cal, test = split_zone(clean)

    base_model = fit_standardized_ridge(train, base_features)
    for part in [train, cal, test]:
        part["base_pred"] = predict_standardized_ridge(part, base_features, base_model)

    residual_train = train.copy()
    residual_train["residual"] = residual_train[TARGET] - residual_train["base_pred"]
    residual_model = fit_standardized_ridge(residual_train, graph_features, target_col="residual", l2=1e-3)
    cal["graph_residual"] = predict_standardized_ridge(cal, graph_features, residual_model)
    test["graph_residual"] = predict_standardized_ridge(test, graph_features, residual_model)
    cal["graph_pred"] = cal["base_pred"] + cal["graph_residual"]
    test["graph_pred"] = test["base_pred"] + test["graph_residual"]

    spike_threshold = float(np.nanquantile(train["lag_1_24_abs"], 0.90))
    rows = []
    daily = test[["timestamp_utc", "zone", TARGET, "lag_1_24_abs", "hour"]].copy()
    daily["local_ridge_pred"] = test["base_pred"]
    daily["graph_temporal_residual_pred"] = test["graph_pred"]
    daily["spike_regime"] = test["lag_1_24_abs"] >= spike_threshold
    daily["spike_threshold"] = spike_threshold

    model_specs = [
        ("Local ridge", "base_pred"),
        ("Graph-temporal residual", "graph_pred"),
    ]
    for regime, mask in [
        ("all", np.ones(len(test), dtype=bool)),
        ("spike", test["lag_1_24_abs"].to_numpy(float) >= spike_threshold),
        ("non_spike", test["lag_1_24_abs"].to_numpy(float) < spike_threshold),
    ]:
        for model_name, col in model_specs:
            sub = test.loc[mask]
            rows.append(
                {
                    "dataset": "OPSD",
                    "zone": zone,
                    "regime": regime,
                    "model": model_name,
                    "spike_threshold": spike_threshold,
                    **point_metrics(sub[TARGET], sub[col]),
                }
            )

    q_global = conformal_quantile(np.abs(cal[TARGET].to_numpy(float) - cal["graph_pred"].to_numpy(float)))
    lower = test["graph_pred"].to_numpy(float) - q_global
    upper = test["graph_pred"].to_numpy(float) + q_global
    interval_row = {
        "dataset": "OPSD",
        "zone": zone,
        "model": "Graph-temporal residual conformal",
        "target_coverage": 1 - ALPHA,
        "calibration_width": 2 * q_global,
        **interval_metrics(test[TARGET], test["graph_pred"], lower, upper),
    }
    daily["graph_lower"] = lower
    daily["graph_upper"] = upper
    return rows, interval_row, daily


def run():
    tidy = prepare_frame()
    rows = []
    interval_rows = []
    daily_parts = []
    for zone, zdf in tidy.groupby("zone", sort=True):
        zone_rows, interval_row, daily = evaluate_zone(zone, zdf)
        rows.extend(zone_rows)
        interval_rows.append(interval_row)
        daily_parts.append(daily)

    point = pd.DataFrame(rows)
    local = point[(point["model"] == "Local ridge")][["zone", "regime", "rmse", "mae"]].rename(
        columns={"rmse": "local_rmse", "mae": "local_mae"}
    )
    point = point.merge(local, on=["zone", "regime"], how="left")
    point["rmse_improvement_pct"] = (point["local_rmse"] - point["rmse"]) / point["local_rmse"] * 100
    point["mae_improvement_pct"] = (point["local_mae"] - point["mae"]) / point["local_mae"] * 100
    interval = pd.DataFrame(interval_rows)
    daily = pd.concat(daily_parts, ignore_index=True)

    point.to_csv(RESULTS / "opsd_graph_temporal_price_ablation_summary.csv", index=False)
    interval.to_csv(RESULTS / "opsd_graph_temporal_price_conformal_summary.csv", index=False)
    daily.to_csv(RESULTS / "opsd_graph_temporal_price_daily.csv", index=False)
    return point, interval, daily


def plot_spike_rmse(point):
    spike = point[(point["regime"] == "spike") & point["model"].isin(["Local ridge", "Graph-temporal residual"])].copy()
    zones = sorted(spike["zone"].unique())
    width, height = 1800, 1080
    left, top, right, bottom = 170, 185, 1660, 855
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "Graph-temporal residual ablation on OPSD spike regimes", fill="#172033", font=font(44, True))
    d.text((88, 112), "Spike subset is defined by each zone's training 90th percentile of |price lag 1 - lag 24|", fill="#5f6b7a", font=font(24))
    ymax = float(spike["rmse"].max() * 1.18)
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        d.line((left, y, right, y), fill="#D7DCE2", width=1)
        val = ymax * i / 5
        d.text((left - 115, y - 13), f"{val:.1f}", fill="#526070", font=font(22))
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    colors = {"Local ridge": "#4C78A8", "Graph-temporal residual": "#54A24B"}
    bar_w = 105
    group_w = (right - left) / len(zones)
    for idx, zone in enumerate(zones):
        base_x = left + idx * group_w + group_w / 2
        for j, model in enumerate(["Local ridge", "Graph-temporal residual"]):
            val = float(spike[(spike["zone"] == zone) & (spike["model"] == model)]["rmse"].iloc[0])
            x0 = base_x + (j - 0.5) * (bar_w + 20)
            x1 = x0 + bar_w
            y0 = bottom - val / ymax * (bottom - top)
            d.rounded_rectangle((x0, y0, x1, bottom), radius=8, fill=colors[model])
            d.text((x0 - 7, y0 - 31), f"{val:.2f}", fill="#1f2937", font=font(21, True))
        tw = d.textlength(zone, font=font(24, True))
        d.text((base_x - tw / 2, bottom + 35), zone, fill="#1f2937", font=font(24, True))
    d.rectangle((330, 930, 364, 958), fill=colors["Local ridge"])
    d.text((378, 924), "Local lag-calendar-exogenous ridge", fill="#526070", font=font(24))
    d.rectangle((880, 930, 914, 958), fill=colors["Graph-temporal residual"])
    d.text((928, 924), "Graph-temporal residual correction", fill="#526070", font=font(24))
    d.text((90, 1005), "Source: OPSD public day-ahead prices with lagged cross-zone graph features; final 20% chronological test split.", fill="#6b7280", font=font(21))
    out = FIGURES / "paper1_fig6_opsd_graph_temporal_spike_rmse.png"
    img.save(out)
    return out


def plot_improvement(point):
    all_rows = point[(point["regime"] == "all") & (point["model"] == "Graph-temporal residual")].copy()
    spike_rows = point[(point["regime"] == "spike") & (point["model"] == "Graph-temporal residual")].copy()
    zones = sorted(all_rows["zone"].unique())
    width, height = 1800, 1080
    left, top, right, bottom = 170, 185, 1660, 855
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "Relative RMSE change from graph-temporal residual correction", fill="#172033", font=font(43, True))
    d.text((88, 112), "Positive values indicate lower RMSE than the local ridge baseline", fill="#5f6b7a", font=font(24))
    vals = list(all_rows["rmse_improvement_pct"]) + list(spike_rows["rmse_improvement_pct"])
    ymin = min(-5, float(min(vals) * 1.25))
    ymax = max(5, float(max(vals) * 1.25))
    zero_y = bottom - (0 - ymin) / (ymax - ymin) * (bottom - top)
    for i in range(7):
        val = ymin + (ymax - ymin) * i / 6
        y = bottom - (val - ymin) / (ymax - ymin) * (bottom - top)
        d.line((left, y, right, y), fill="#D7DCE2", width=1)
        d.text((left - 115, y - 13), f"{val:.1f}%", fill="#526070", font=font(22))
    d.line((left, zero_y, right, zero_y), fill="#111827", width=3)
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    colors = {"all": "#72B7B2", "spike": "#E45756"}
    bar_w = 105
    group_w = (right - left) / len(zones)
    for idx, zone in enumerate(zones):
        base_x = left + idx * group_w + group_w / 2
        for j, regime in enumerate(["all", "spike"]):
            frame = all_rows if regime == "all" else spike_rows
            val = float(frame[frame["zone"] == zone]["rmse_improvement_pct"].iloc[0])
            x0 = base_x + (j - 0.5) * (bar_w + 20)
            x1 = x0 + bar_w
            yv = bottom - (val - ymin) / (ymax - ymin) * (bottom - top)
            y0, y1 = (yv, zero_y) if val >= 0 else (zero_y, yv)
            d.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=colors[regime])
            d.text((x0 - 10, min(y0, y1) - 31), f"{val:.1f}%", fill="#1f2937", font=font(21, True))
        tw = d.textlength(zone, font=font(24, True))
        d.text((base_x - tw / 2, bottom + 35), zone, fill="#1f2937", font=font(24, True))
    d.rectangle((445, 930, 479, 958), fill=colors["all"])
    d.text((493, 924), "All test hours", fill="#526070", font=font(24))
    d.rectangle((770, 930, 804, 958), fill=colors["spike"])
    d.text((818, 924), "Spike-regime hours", fill="#526070", font=font(24))
    d.text((90, 1005), "Source: OPSD public day-ahead prices; graph features use only lagged cross-zone prices.", fill="#6b7280", font=font(21))
    out = FIGURES / "paper1_fig7_opsd_graph_temporal_improvement.png"
    img.save(out)
    return out


def main():
    point, interval, _ = run()
    fig1 = plot_spike_rmse(point)
    fig2 = plot_improvement(point)
    print(RESULTS / "opsd_graph_temporal_price_ablation_summary.csv")
    print(RESULTS / "opsd_graph_temporal_price_conformal_summary.csv")
    print(RESULTS / "opsd_graph_temporal_price_daily.csv")
    print(fig1)
    print(fig2)
    print(point.to_string(index=False))
    print(interval.to_string(index=False))


if __name__ == "__main__":
    main()
