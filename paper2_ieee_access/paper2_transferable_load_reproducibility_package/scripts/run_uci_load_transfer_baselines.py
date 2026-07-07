from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
RAW_ZIP = ROOT / "data" / "raw" / "electricityloaddiagrams20112014.zip"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

DATA_FILE = "LD2011_2014.txt"
YEAR_START = pd.Timestamp("2014-01-01 00:00:00")
YEAR_END = pd.Timestamp("2015-01-01 00:00:00")
TEST_START = pd.Timestamp("2014-10-01 00:00:00")
ADAPT_7_START = TEST_START - pd.Timedelta(days=7)
ADAPT_28_START = TEST_START - pd.Timedelta(days=28)
SOURCE_TRAIN_START = pd.Timestamp("2014-01-08 00:00:00")
N_SOURCE = 30
N_TARGET = 10


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


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = y - pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "smape": smape(y, pred),
        "n": int(len(y)),
    }


def fit_linear(train, feature_cols, target_col="load_norm", l2=1e-5):
    x = train[feature_cols].to_numpy(float)
    y = train[target_col].to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    eye = np.eye(x.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + l2 * eye, x.T @ y)


def predict_linear(frame, feature_cols, beta):
    x = frame[feature_cols].to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    return x @ beta


def read_columns():
    with zipfile.ZipFile(RAW_ZIP) as zf:
        with zf.open(DATA_FILE) as fh:
            cols = pd.read_csv(fh, sep=";", nrows=0).columns.tolist()
    return [c.strip('"') for c in cols]


def scan_2014_client_stats():
    rows = []
    with zipfile.ZipFile(RAW_ZIP) as zf:
        with zf.open(DATA_FILE) as fh:
            reader = pd.read_csv(fh, sep=";", decimal=",", chunksize=5000)
            for chunk in reader:
                time_col = chunk.columns[0]
                chunk = chunk.rename(columns={time_col: "timestamp"})
                chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
                chunk = chunk[(chunk["timestamp"] >= YEAR_START) & (chunk["timestamp"] < YEAR_END)]
                if chunk.empty:
                    continue
                numeric = chunk.drop(columns=["timestamp"]).apply(pd.to_numeric, errors="coerce")
                for col in numeric.columns:
                    series = numeric[col]
                    valid = series.dropna()
                    rows.append(
                        {
                            "client": col,
                            "valid": int(series.notna().sum()),
                            "nonzero": int((series.fillna(0) > 0).sum()),
                            "sum": float(valid.sum()),
                            "sum_sq": float((valid**2).sum()),
                        }
                    )
    stats = pd.DataFrame(rows).groupby("client", as_index=False).sum(numeric_only=True)
    stats["mean"] = stats["sum"] / stats["valid"].replace(0, np.nan)
    numerator = stats["sum_sq"] - (stats["sum"] ** 2) / stats["valid"].replace(0, np.nan)
    stats["std"] = np.sqrt((numerator / (stats["valid"] - 1).replace(0, np.nan)).clip(lower=0))
    stats["coverage"] = stats["valid"] / (365 * 24 * 4)
    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
    stats = stats[(stats["coverage"] >= 0.95) & (stats["mean"] > 1) & (stats["std"] > 1)].copy()
    stats = stats.sort_values(["coverage", "std", "mean", "client"], ascending=[False, False, False, True])
    stats.to_csv(RESULTS / "uci_load_client_selection_stats.csv", index=False)
    return stats


def load_selected_hourly(selected_clients):
    all_cols = read_columns()
    usecols = [all_cols[0]] + selected_clients
    with zipfile.ZipFile(RAW_ZIP) as zf:
        with zf.open(DATA_FILE) as fh:
            raw = pd.read_csv(fh, sep=";", decimal=",", usecols=usecols)
    time_col = raw.columns[0]
    raw = raw.rename(columns={time_col: "timestamp"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw[(raw["timestamp"] >= YEAR_START) & (raw["timestamp"] < YEAR_END)].copy()
    raw = raw.set_index("timestamp")
    raw = raw.apply(pd.to_numeric, errors="coerce")
    hourly = raw.resample("1h").mean().sort_index()
    hourly = hourly.interpolate(limit=3).ffill(limit=3).bfill(limit=3)
    hourly.to_csv(PROCESSED / "uci_electricity_hourly_selected_clients_wide.csv")
    long = hourly.reset_index().melt(id_vars="timestamp", var_name="client", value_name="load")
    long.to_csv(PROCESSED / "uci_electricity_hourly_selected_clients.csv", index=False)
    return hourly


def make_supervised(hourly, source_clients, target_clients):
    long = hourly.reset_index().melt(id_vars="timestamp", var_name="client", value_name="load")
    long = long.sort_values(["client", "timestamp"])
    for lag in [1, 24, 168]:
        long[f"lag_{lag}"] = long.groupby("client")["load"].shift(lag)
    long["hour"] = long["timestamp"].dt.hour
    long["dayofweek"] = long["timestamp"].dt.dayofweek
    long["month"] = long["timestamp"].dt.month
    long["hour_sin"] = np.sin(2 * np.pi * long["hour"] / 24)
    long["hour_cos"] = np.cos(2 * np.pi * long["hour"] / 24)
    long["dow_sin"] = np.sin(2 * np.pi * long["dayofweek"] / 7)
    long["dow_cos"] = np.cos(2 * np.pi * long["dayofweek"] / 7)

    scale = {}
    for client in source_clients:
        train = long[
            (long["client"] == client)
            & (long["timestamp"] >= SOURCE_TRAIN_START)
            & (long["timestamp"] < TEST_START)
        ]
        scale[client] = max(float(train["load"].median()), 1e-6)
    for client in target_clients:
        adapt = long[
            (long["client"] == client)
            & (long["timestamp"] >= ADAPT_28_START)
            & (long["timestamp"] < TEST_START)
        ]
        scale[client] = max(float(adapt["load"].median()), 1e-6)

    long["scale"] = long["client"].map(scale)
    long["load_norm"] = long["load"] / long["scale"]
    for lag in [1, 24, 168]:
        long[f"lag_{lag}_norm"] = long[f"lag_{lag}"] / long["scale"]
    long = long.dropna(subset=["load", "lag_1", "lag_24", "lag_168", "scale"]).copy()
    return long


def evaluate(hourly, source_clients, target_clients):
    data = make_supervised(hourly, source_clients, target_clients)
    feature_cols = [
        "lag_1_norm",
        "lag_24_norm",
        "lag_168_norm",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month",
    ]
    source_train = data[
        data["client"].isin(source_clients)
        & (data["timestamp"] >= SOURCE_TRAIN_START)
        & (data["timestamp"] < TEST_START)
    ].copy()
    rows = []

    beta_source = fit_linear(source_train, feature_cols)

    for client in target_clients:
        cdf = data[data["client"] == client].copy()
        test = cdf[cdf["timestamp"] >= TEST_START].copy()
        if len(test) < 24 * 30:
            continue
        for lag, name in [(24, "Seasonal-24h"), (168, "Seasonal-168h")]:
            m = metrics(test["load"], test[f"lag_{lag}"])
            rows.append({"dataset": "UCI Electricity Load Diagrams 2011-2014", "target_client": client, "model": name, "protocol": "direct seasonal", **m})

        pred = predict_linear(test, feature_cols, beta_source) * test["scale"].to_numpy(float)
        rows.append({"dataset": "UCI Electricity Load Diagrams 2011-2014", "target_client": client, "model": "Pooled-source-linear", "protocol": "source-only transfer", **metrics(test["load"], pred)})

        for days, start in [(7, ADAPT_7_START), (28, ADAPT_28_START)]:
            adapt = cdf[(cdf["timestamp"] >= start) & (cdf["timestamp"] < TEST_START)].copy()
            if len(adapt) < 24 * days * 0.8:
                continue

            beta_target = fit_linear(adapt, feature_cols)
            pred = predict_linear(test, feature_cols, beta_target) * test["scale"].to_numpy(float)
            rows.append({"dataset": "UCI Electricity Load Diagrams 2011-2014", "target_client": client, "model": f"Target-{days}d-linear", "protocol": "few-shot target only", **metrics(test["load"], pred)})

            combo = pd.concat([source_train, adapt], ignore_index=True)
            beta_combo = fit_linear(combo, feature_cols)
            pred = predict_linear(test, feature_cols, beta_combo) * test["scale"].to_numpy(float)
            rows.append({"dataset": "UCI Electricity Load Diagrams 2011-2014", "target_client": client, "model": f"Pooled+target-{days}d-linear", "protocol": "few-shot transfer", **metrics(test["load"], pred)})

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "uci_load_transfer_baselines.csv", index=False)
    summary = (
        result.groupby(["model", "protocol"], as_index=False)
        .agg(target_clients=("target_client", "nunique"), mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_smape=("smape", "mean"), total_n=("n", "sum"))
        .sort_values("mean_rmse")
    )
    summary.to_csv(RESULTS / "uci_load_transfer_summary.csv", index=False)
    return result, summary


def write_dataset_stats(stats, source_clients, target_clients):
    selected = stats[stats["client"].isin(source_clients + target_clients)].copy()
    selected["role"] = np.where(selected["client"].isin(source_clients), "source", "target")
    selected.to_csv(RESULTS / "uci_load_dataset_stats.csv", index=False)


def plot_summary(summary):
    plot = summary.sort_values("mean_rmse", ascending=True).copy()
    width, height = 1800, 1050
    left, top, right, bottom = 560, 170, 1660, 900
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "UCI multi-client load transfer benchmark", fill="#172033", font=font(46, True))
    d.text((92, 112), "Mean RMSE across ten target clients in the 2014 public load dataset", fill="#5f6b7a", font=font(25))
    max_v = float(plot["mean_rmse"].max()) * 1.12
    for i in range(6):
        x = left + i * (right - left) / 5
        d.line((x, top, x, bottom), fill="#D7DCE2", width=1)
        label = f"{max_v * i / 5:.0f}"
        d.text((x - d.textlength(label, font=font(21)) / 2, bottom + 20), label, fill="#526070", font=font(21))
    row_h = (bottom - top) / len(plot)
    for idx, row in enumerate(plot.itertuples(index=False)):
        y = top + idx * row_h + row_h * 0.18
        bar_h = row_h * 0.58
        model = row.model
        color = "#4C78A8" if "Pooled" in model else "#F58518" if "Target" in model else "#54A24B"
        x1 = left + float(row.mean_rmse) / max_v * (right - left)
        d.rounded_rectangle((left, y, x1, y + bar_h), radius=8, fill=color)
        d.text((90, y + 5), model, fill="#1f2937", font=font(24, True if idx == 0 else False))
        val = f"{float(row.mean_rmse):.2f}"
        d.text((x1 + 14, y + 4), val, fill="#1f2937", font=font(23, True))
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    d.text((left, 970), "Lower is better. Linear transfer baselines are transparent first-layer evidence, not the final SSL model.", fill="#526070", font=font(23))
    out = FIGURES / "paper2_fig4_uci_transfer_rmse.png"
    img.save(out)
    return out


def main():
    if not RAW_ZIP.exists():
        raise SystemExit(f"Missing UCI raw zip: {RAW_ZIP}")
    stats = scan_2014_client_stats()
    selected = stats.head(N_SOURCE + N_TARGET)["client"].tolist()
    source_clients = selected[:N_SOURCE]
    target_clients = selected[N_SOURCE:]
    hourly = load_selected_hourly(selected)
    result, summary = evaluate(hourly, source_clients, target_clients)
    write_dataset_stats(stats, source_clients, target_clients)
    figure = plot_summary(summary)
    print(PROCESSED / "uci_electricity_hourly_selected_clients.csv")
    print(RESULTS / "uci_load_client_selection_stats.csv")
    print(RESULTS / "uci_load_dataset_stats.csv")
    print(RESULTS / "uci_load_transfer_baselines.csv")
    print(RESULTS / "uci_load_transfer_summary.csv")
    print(figure)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
