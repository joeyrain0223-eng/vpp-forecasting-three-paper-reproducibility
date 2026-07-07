from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from run_uci_appliances_energy_baselines import (
    FIGURES,
    LOOKBACK_STEPS,
    RANDOM_FILTERS,
    RESULTS,
    add_random_window_features,
    fit_ridge,
    font,
    load_raw,
    mae,
    predict_ridge,
    rmse,
    smape,
    split_data,
)


OUT_RESULTS = RESULTS / "uci_appliances_energy_multihorizon_robustness.csv"
OUT_SUMMARY = RESULTS / "uci_appliances_energy_multihorizon_summary.csv"
OUT_FIGURE = FIGURES / "paper2_fig13_uci_appliances_multihorizon_robustness.png"

HORIZONS = [
    ("1h", 6),
    ("3h", 18),
    ("6h", 36),
    ("12h", 72),
]


def make_supervised(df: pd.DataFrame, horizon_label: str, horizon_steps: int) -> pd.DataFrame:
    out = df.copy()
    out["horizon"] = horizon_label
    out["target_appliances"] = out["Appliances"].shift(-horizon_steps)
    out["target_timestamp"] = out["timestamp"] + pd.to_timedelta(horizon_steps * 10, unit="min")
    out["lag_now"] = out["Appliances"]
    out["lag_1h"] = out["Appliances"].shift(6)
    out["lag_24h_same_target_slot"] = out["Appliances"].shift(LOOKBACK_STEPS - horizon_steps)
    out["rolling_1h_mean"] = out["Appliances"].shift(1).rolling(6).mean()
    out["rolling_24h_mean"] = out["Appliances"].shift(1).rolling(LOOKBACK_STEPS).mean()
    out["current_hour"] = out["timestamp"].dt.hour
    out["current_dayofweek"] = out["timestamp"].dt.dayofweek
    out["target_hour"] = out["target_timestamp"].dt.hour
    out["target_dayofweek"] = out["target_timestamp"].dt.dayofweek
    out["current_hour_sin"] = np.sin(2 * np.pi * out["current_hour"] / 24)
    out["current_hour_cos"] = np.cos(2 * np.pi * out["current_hour"] / 24)
    out["target_hour_sin"] = np.sin(2 * np.pi * out["target_hour"] / 24)
    out["target_hour_cos"] = np.cos(2 * np.pi * out["target_hour"] / 24)
    out["target_dow_sin"] = np.sin(2 * np.pi * out["target_dayofweek"] / 7)
    out["target_dow_cos"] = np.cos(2 * np.pi * out["target_dayofweek"] / 7)
    return out.dropna().reset_index(drop=True)


def metric_row(horizon: str, model: str, protocol: str, frame: pd.DataFrame, pred: np.ndarray) -> dict:
    y = frame["target_appliances"].to_numpy(float)
    return {
        "dataset": "UCI Appliances Energy Prediction",
        "horizon": horizon,
        "model": model,
        "protocol": protocol,
        "mae_wh": mae(y, pred),
        "rmse_wh": rmse(y, pred),
        "smape": smape(y, pred),
        "n": int(len(y)),
    }


def evaluate_horizon(raw: pd.DataFrame, horizon_label: str, horizon_steps: int) -> list[dict]:
    supervised = make_supervised(raw, horizon_label, horizon_steps)
    with_random, random_cols = add_random_window_features(supervised.copy())
    train, _, test = split_data(with_random)

    base_features = [
        "lag_now",
        "lag_1h",
        "lag_24h_same_target_slot",
        "rolling_1h_mean",
        "rolling_24h_mean",
        "T_out",
        "RH_out",
        "Windspeed",
        "Visibility",
        "Tdewpoint",
        "current_hour_sin",
        "current_hour_cos",
        "target_hour_sin",
        "target_hour_cos",
        "target_dow_sin",
        "target_dow_cos",
    ]

    rows = [
        metric_row(horizon_label, "Persistence-current", "current-value persistence", test, test["lag_now"].to_numpy(float)),
        metric_row(
            horizon_label,
            "Seasonal-24h-target-slot",
            "previous-day same target-slot seasonal",
            test,
            test["lag_24h_same_target_slot"].to_numpy(float),
        ),
    ]

    lag_model = fit_ridge(train, base_features, "target_appliances", l2=1e-3)
    lag_pred = predict_ridge(test, base_features, lag_model)
    rows.append(metric_row(horizon_label, "Lag-weather ridge", "chronological ridge baseline", test, lag_pred))

    random_features = base_features + random_cols
    random_model = fit_ridge(train, random_features, "target_appliances", l2=1e-2)
    random_pred = predict_ridge(test, random_features, random_model)
    rows.append(
        metric_row(
            horizon_label,
            "Random-window ridge",
            "deterministic random temporal representation",
            test,
            random_pred,
        )
    )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    best_by_horizon = results.sort_values(["horizon", "rmse_wh"]).groupby("horizon", as_index=False).first()
    pivot = results.pivot(index="horizon", columns="model", values="rmse_wh").reset_index()
    rows = []
    for _, row in pivot.iterrows():
        horizon = row["horizon"]
        best = best_by_horizon[best_by_horizon["horizon"] == horizon].iloc[0]
        rows.append(
            {
                "horizon": horizon,
                "best_model": best["model"],
                "best_rmse_wh": float(best["rmse_wh"]),
                "lag_weather_rmse_wh": float(row["Lag-weather ridge"]),
                "random_window_rmse_wh": float(row["Random-window ridge"]),
                "seasonal_rmse_wh": float(row["Seasonal-24h-target-slot"]),
                "persistence_rmse_wh": float(row["Persistence-current"]),
                "lag_weather_gain_vs_persistence_pct": (float(row["Persistence-current"]) - float(row["Lag-weather ridge"]))
                / float(row["Persistence-current"])
                * 100,
                "random_window_gain_vs_lag_weather_pct": (float(row["Lag-weather ridge"]) - float(row["Random-window ridge"]))
                / float(row["Lag-weather ridge"])
                * 100,
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame) -> Path:
    summary = summary.copy()
    horizon_order = {label: i for i, (label, _) in enumerate(HORIZONS)}
    summary["order"] = summary["horizon"].map(horizon_order)
    summary = summary.sort_values("order")

    width, height = 1850, 1050
    left, top, right, bottom = 230, 190, 1650, 770
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 55), "UCI Appliances multi-horizon robustness", fill="#172033", font=font(46, True))
    d.text(
        (82, 112),
        "Chronological public holdout; lower RMSE is better; horizons use current information only",
        fill="#5f6b7a",
        font=font(25),
    )

    max_v = float(
        summary[["lag_weather_rmse_wh", "random_window_rmse_wh", "seasonal_rmse_wh", "persistence_rmse_wh"]].to_numpy().max()
    ) * 1.12
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        d.line((left, y, right, y), fill="#D7DCE2", width=1)
        label = f"{max_v * i / 5:.0f}"
        d.text((70, y - 13), label, fill="#526070", font=font(21))

    colors = {
        "lag_weather_rmse_wh": "#4C78A8",
        "random_window_rmse_wh": "#54A24B",
        "seasonal_rmse_wh": "#F58518",
        "persistence_rmse_wh": "#B279A2",
    }
    labels = {
        "lag_weather_rmse_wh": "Lag-weather",
        "random_window_rmse_wh": "Random-window",
        "seasonal_rmse_wh": "Seasonal",
        "persistence_rmse_wh": "Persistence",
    }
    group_w = (right - left) / len(summary)
    bar_w = group_w / 6
    series = list(colors)
    for hidx, row in enumerate(summary.itertuples(index=False)):
        center = left + group_w * hidx + group_w / 2
        x0 = center - (len(series) * bar_w) / 2
        for sidx, col in enumerate(series):
            val = float(getattr(row, col))
            x = x0 + sidx * bar_w
            y = bottom - val / max_v * (bottom - top)
            d.rounded_rectangle((x, y, x + bar_w * 0.78, bottom), radius=5, fill=colors[col])
        d.text((center - 22, bottom + 28), row.horizon, fill="#1f2937", font=font(25, True))

    legend_x = 1260
    for idx, col in enumerate(series):
        y = 810 + idx * 38
        d.rounded_rectangle((legend_x, y, legend_x + 28, y + 22), radius=4, fill=colors[col])
        d.text((legend_x + 42, y - 4), labels[col], fill="#344054", font=font(23))
    d.text(
        (80, 985),
        "Interpretation: the transparent lag-weather model remains stable across horizons; random temporal filters do not dominate this external dataset.",
        fill="#526070",
        font=font(23),
    )
    img.save(OUT_FIGURE)
    return OUT_FIGURE


def main() -> None:
    raw = load_raw()
    rows = []
    for horizon_label, horizon_steps in HORIZONS:
        rows.extend(evaluate_horizon(raw, horizon_label, horizon_steps))
    results = pd.DataFrame(rows).sort_values(["horizon", "rmse_wh", "model"])
    summary = summarize(results)
    results.to_csv(OUT_RESULTS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    figure = plot_summary(summary)
    print(OUT_RESULTS)
    print(OUT_SUMMARY)
    print(figure)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
