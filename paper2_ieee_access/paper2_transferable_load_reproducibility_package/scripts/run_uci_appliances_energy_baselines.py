from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

UCI_URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
RAW_ZIP = RAW_DIR / "uci_appliances_energy_prediction.zip"
RAW_CSV_NAME = "energydata_complete.csv"
HORIZON_STEPS = 6  # 10-minute data, six steps is one hour ahead.
LOOKBACK_STEPS = 144  # 24 hours.
RANDOM_FILTERS = 48


def font(size: int = 28, bold: bool = False):
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


def download_if_needed() -> None:
    if RAW_ZIP.exists() and RAW_ZIP.stat().st_size > 0:
        return
    with urllib.request.urlopen(UCI_URL, timeout=60) as response:
        RAW_ZIP.write_bytes(response.read())


def load_raw() -> pd.DataFrame:
    download_if_needed()
    with zipfile.ZipFile(RAW_ZIP) as zf:
        with zf.open(RAW_CSV_NAME) as fh:
            df = pd.read_csv(fh)
    df["timestamp"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["date"]).sort_values("timestamp").reset_index(drop=True)
    return df


def make_supervised(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_appliances_1h"] = out["Appliances"].shift(-HORIZON_STEPS)
    out["lag_now"] = out["Appliances"]
    out["lag_1h"] = out["Appliances"].shift(HORIZON_STEPS)
    out["lag_24h_same_horizon"] = out["Appliances"].shift(LOOKBACK_STEPS - HORIZON_STEPS)
    out["rolling_1h_mean"] = out["Appliances"].shift(1).rolling(HORIZON_STEPS).mean()
    out["rolling_24h_mean"] = out["Appliances"].shift(1).rolling(LOOKBACK_STEPS).mean()
    out["hour"] = out["timestamp"].dt.hour
    out["dayofweek"] = out["timestamp"].dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out = out.dropna().reset_index(drop=True)
    out.to_csv(PROCESSED / "uci_appliances_energy_supervised_1h.csv", index=False)
    return out


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(data)
    train_end = int(n * 0.60)
    valid_end = int(n * 0.80)
    return data.iloc[:train_end].copy(), data.iloc[train_end:valid_end].copy(), data.iloc[valid_end:].copy()


def rmse(y, pred) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y, pred) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(y - pred)))


def smape(y, pred) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    den = np.abs(y) + np.abs(pred)
    score = np.zeros_like(den, dtype=float)
    mask = den > 0
    score[mask] = 2 * np.abs(y[mask] - pred[mask]) / den[mask]
    return float(np.mean(score) * 100)


def metric_row(model: str, protocol: str, frame: pd.DataFrame, pred: np.ndarray) -> dict:
    y = frame["target_appliances_1h"].to_numpy(float)
    return {
        "dataset": "UCI Appliances Energy Prediction",
        "model": model,
        "protocol": protocol,
        "mae_wh": mae(y, pred),
        "rmse_wh": rmse(y, pred),
        "smape": smape(y, pred),
        "n": int(len(y)),
    }


def fit_ridge(train: pd.DataFrame, feature_cols: list[str], target_col: str, l2: float = 1e-3) -> np.ndarray:
    x = train[feature_cols].to_numpy(float)
    y = train[target_col].to_numpy(float)
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1.0
    xz = (x - means) / stds
    xz = np.column_stack([np.ones(len(xz)), xz])
    eye = np.eye(xz.shape[1])
    eye[0, 0] = 0
    beta = np.linalg.solve(xz.T @ xz + l2 * eye, xz.T @ y)
    return np.r_[beta, means, stds]


def predict_ridge(frame: pd.DataFrame, feature_cols: list[str], packed: np.ndarray) -> np.ndarray:
    p = len(feature_cols)
    beta = packed[: p + 1]
    means = packed[p + 1 : p + 1 + p]
    stds = packed[p + 1 + p :]
    x = frame[feature_cols].to_numpy(float)
    xz = (x - means) / stds
    xz = np.column_stack([np.ones(len(xz)), xz])
    return xz @ beta


def add_random_window_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    series = data["Appliances"].to_numpy(float)
    idx = np.arange(len(data))
    valid = idx >= LOOKBACK_STEPS
    windows = np.lib.stride_tricks.sliding_window_view(series, LOOKBACK_STEPS)
    aligned = np.full((len(data), LOOKBACK_STEPS), np.nan)
    aligned[LOOKBACK_STEPS - 1 :] = windows
    rng = np.random.default_rng(20260630)
    filters = rng.normal(size=(RANDOM_FILTERS, LOOKBACK_STEPS))
    filters = filters / np.linalg.norm(filters, axis=1, keepdims=True)
    feature_names = []
    for j, filt in enumerate(filters):
        response = np.full(len(data), np.nan)
        response[valid] = aligned[valid] @ filt
        data[f"rw_{j:02d}_activation"] = response
        feature_names.append(f"rw_{j:02d}_activation")
    return data.dropna().reset_index(drop=True), feature_names


def evaluate(supervised: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    with_random, random_cols = add_random_window_features(supervised.copy())
    train, valid, test = split_data(with_random)

    base_features = [
        "lag_now",
        "lag_1h",
        "lag_24h_same_horizon",
        "rolling_1h_mean",
        "rolling_24h_mean",
        "T_out",
        "RH_out",
        "Windspeed",
        "Visibility",
        "Tdewpoint",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    rows = [
        metric_row("Persistence-current", "one-hour-ahead public holdout", test, test["lag_now"].to_numpy(float)),
        metric_row("Seasonal-24h", "one-hour-ahead public holdout", test, test["lag_24h_same_horizon"].to_numpy(float)),
    ]

    lag_model = fit_ridge(train, base_features, "target_appliances_1h")
    lag_pred = predict_ridge(test, base_features, lag_model)
    rows.append(metric_row("Lag-weather ridge", "chronological ridge baseline", test, lag_pred))

    random_features = base_features + random_cols
    random_model = fit_ridge(train, random_features, "target_appliances_1h", l2=1e-2)
    random_pred = predict_ridge(test, random_features, random_model)
    rows.append(metric_row("Random-window ridge", "deterministic random temporal representation", test, random_pred))

    results = pd.DataFrame(rows).sort_values("rmse_wh")
    results.to_csv(RESULTS / "uci_appliances_energy_baselines.csv", index=False)

    stats = pd.DataFrame(
        [
            {
                "dataset": "UCI Appliances Energy Prediction",
                "rows_raw": int(len(load_raw())),
                "rows_supervised": int(len(supervised)),
                "rows_modelled_after_window": int(len(with_random)),
                "train_rows": int(len(train)),
                "validation_rows": int(len(valid)),
                "test_rows": int(len(test)),
                "start": str(with_random["timestamp"].min()),
                "end": str(with_random["timestamp"].max()),
                "target": "Appliances one hour ahead",
                "source_url": UCI_URL,
            }
        ]
    )
    stats.to_csv(RESULTS / "uci_appliances_energy_dataset_stats.csv", index=False)
    return results, stats


def plot_results(results: pd.DataFrame) -> Path:
    plot = results.sort_values("rmse_wh", ascending=True).copy()
    width, height = 1700, 950
    left, top, right, bottom = 470, 170, 1520, 760
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 55), "Second public load dataset: UCI Appliances Energy", fill="#172033", font=font(44, True))
    d.text((82, 110), "One-hour-ahead appliance-load RMSE on chronological public holdout", fill="#5f6b7a", font=font(25))
    max_v = float(plot["rmse_wh"].max()) * 1.15
    for i in range(6):
        x = left + i * (right - left) / 5
        d.line((x, top, x, bottom), fill="#D7DCE2", width=1)
        label = f"{max_v * i / 5:.0f}"
        d.text((x - d.textlength(label, font=font(21)) / 2, bottom + 20), label, fill="#526070", font=font(21))
    row_h = (bottom - top) / len(plot)
    for idx, row in enumerate(plot.itertuples(index=False)):
        y = top + idx * row_h + row_h * 0.20
        bar_h = row_h * 0.56
        color = "#4C78A8" if "Random" in row.model else "#54A24B" if "ridge" in row.model else "#F58518"
        x1 = left + float(row.rmse_wh) / max_v * (right - left)
        d.rounded_rectangle((left, y, x1, y + bar_h), radius=8, fill=color)
        d.text((80, y + 7), row.model, fill="#1f2937", font=font(25, idx == 0))
        val = f"{float(row.rmse_wh):.1f}"
        d.text((x1 + 14, y + 6), val, fill="#1f2937", font=font(24, True))
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    d.text((left, 835), "Lower is better. The result is an external public sanity check, not a replacement for multi-client transfer evidence.", fill="#526070", font=font(22))
    out = FIGURES / "paper2_fig10_uci_appliances_second_dataset.png"
    img.save(out)
    return out


def main() -> None:
    raw = load_raw()
    supervised = make_supervised(raw)
    results, stats = evaluate(supervised)
    figure = plot_results(results)
    print(RAW_ZIP)
    print(PROCESSED / "uci_appliances_energy_supervised_1h.csv")
    print(RESULTS / "uci_appliances_energy_dataset_stats.csv")
    print(RESULTS / "uci_appliances_energy_baselines.csv")
    print(figure)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
