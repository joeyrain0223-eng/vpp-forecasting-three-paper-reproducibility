from pathlib import Path
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
RAW = ROOT / "data" / "raw" / "opsd_time_series_60min_singleindex.csv"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

ZONES = {
    "DE_LU": {
        "price": "DE_LU_price_day_ahead",
        "load": "DE_LU_load_actual_entsoe_transparency",
        "solar": "DE_LU_solar_generation_actual",
        "wind": "DE_LU_wind_generation_actual",
    },
    "DK_1": {
        "price": "DK_1_price_day_ahead",
        "load": "DK_1_load_actual_entsoe_transparency",
        "solar": "DK_1_solar_generation_actual",
        "wind": "DK_1_wind_generation_actual",
    },
    "DK_2": {
        "price": "DK_2_price_day_ahead",
        "load": "DK_2_load_actual_entsoe_transparency",
        "solar": "DK_2_solar_generation_actual",
        "wind": "DK_2_wind_generation_actual",
    },
    "GB_GBN": {
        "price": "GB_GBN_price_day_ahead",
        "load": "GB_GBN_load_actual_entsoe_transparency",
        "solar": "GB_GBN_solar_generation_actual",
        "wind": "GB_GBN_wind_generation_actual",
    },
}


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
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "smape": smape(y, pred),
        "n": int(len(y)),
    }


def fit_linear(train, feature_cols, target_col, l2=1e-6):
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


def prepare_tidy():
    needed = ["utc_timestamp"]
    for cfg in ZONES.values():
        needed.extend(cfg.values())
    raw = pd.read_csv(RAW, usecols=list(dict.fromkeys(needed)))
    raw["timestamp_utc"] = pd.to_datetime(raw["utc_timestamp"], utc=True)
    parts = []
    for zone, cfg in ZONES.items():
        part = pd.DataFrame(
            {
                "timestamp_utc": raw["timestamp_utc"],
                "zone": zone,
                "price_eur_mwh": raw[cfg["price"]],
                "load_mw": raw[cfg["load"]],
                "solar_mw": raw[cfg["solar"]],
                "wind_mw": raw[cfg["wind"]],
            }
        )
        part = part.dropna(subset=["price_eur_mwh", "load_mw"])
        parts.append(part)
    tidy = pd.concat(parts, ignore_index=True)
    tidy = tidy.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    tidy["hour"] = tidy["timestamp_utc"].dt.hour
    tidy["dayofweek"] = tidy["timestamp_utc"].dt.dayofweek
    tidy["month"] = tidy["timestamp_utc"].dt.month
    tidy["hour_sin"] = np.sin(2 * np.pi * tidy["hour"] / 24)
    tidy["hour_cos"] = np.cos(2 * np.pi * tidy["hour"] / 24)
    tidy["dow_sin"] = np.sin(2 * np.pi * tidy["dayofweek"] / 7)
    tidy["dow_cos"] = np.cos(2 * np.pi * tidy["dayofweek"] / 7)
    for col in ["price_eur_mwh", "load_mw", "solar_mw", "wind_mw"]:
        tidy[col] = pd.to_numeric(tidy[col], errors="coerce")
    tidy.to_csv(PROCESSED / "opsd_hourly_price_load_renewables_tidy.csv", index=False)
    return tidy


def add_lags(df, target_col):
    work = df.copy()
    for lag in [1, 24, 168]:
        work[f"{target_col}_lag_{lag}"] = work.groupby("zone")[target_col].shift(lag)
    return work


def evaluate_target(tidy, target_col, task):
    work = add_lags(tidy, target_col)
    rows = []
    feature_cols = [
        f"{target_col}_lag_1",
        f"{target_col}_lag_24",
        f"{target_col}_lag_168",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month",
    ]
    if target_col == "price_eur_mwh":
        feature_cols += ["load_mw", "solar_mw", "wind_mw"]

    for zone, zdf in work.groupby("zone", sort=True):
        zdf = zdf.sort_values("timestamp_utc").reset_index(drop=True)
        n_total = len(zdf)
        test_start = int(n_total * 0.80)
        train = zdf.iloc[:test_start].copy()
        test = zdf.iloc[test_start:].copy()

        for lag, name in [(1, "Persist-1h"), (24, "Seasonal-24h"), (168, "Seasonal-168h")]:
            col = f"{target_col}_lag_{lag}"
            eval_df = test[[target_col, col]].dropna()
            if len(eval_df) < 24:
                continue
            m = metrics(eval_df[target_col], eval_df[col])
            rows.append({"task": task, "zone": zone, "model": name, **m})

        train_l = train[[target_col] + feature_cols].dropna()
        test_l = test[[target_col] + feature_cols].dropna()
        if len(train_l) > 1000 and len(test_l) > 100:
            beta = fit_linear(train_l, feature_cols, target_col)
            pred = predict_linear(test_l, feature_cols, beta)
            m = metrics(test_l[target_col], pred)
            rows.append({"task": task, "zone": zone, "model": "Linear-lag-calendar-exog", **m})
    return rows


def build_stats(tidy):
    rows = []
    for zone, zdf in tidy.groupby("zone", sort=True):
        rows.append(
            {
                "dataset": "OPSD time_series_60min_singleindex",
                "zone": zone,
                "rows": int(len(zdf)),
                "start_utc": zdf["timestamp_utc"].min().isoformat(),
                "end_utc": zdf["timestamp_utc"].max().isoformat(),
                "price_non_null": int(zdf["price_eur_mwh"].notna().sum()),
                "load_non_null": int(zdf["load_mw"].notna().sum()),
                "solar_non_null": int(zdf["solar_mw"].notna().sum()),
                "wind_non_null": int(zdf["wind_mw"].notna().sum()),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(RESULTS / "opsd_public_dataset_stats.csv", index=False)
    return stats


def daily_storage_revenue(prices, charge_n=4, discharge_n=4, efficiency=0.92):
    prices = np.asarray(prices, dtype=float)
    if len(prices) < charge_n + discharge_n:
        return np.nan
    charge_idx = np.argsort(prices)[:charge_n]
    discharge_idx = np.argsort(prices)[-discharge_n:]
    return float(prices[discharge_idx].sum() * efficiency - prices[charge_idx].sum() / efficiency)


def evaluate_vpp_decision(tidy):
    rows = []
    daily_rows = []
    price = tidy[["timestamp_utc", "zone", "price_eur_mwh"]].dropna().copy()
    price["date"] = price["timestamp_utc"].dt.date
    for zone, zdf in price.groupby("zone", sort=True):
        days = []
        for date, day in zdf.groupby("date", sort=True):
            if len(day) < 20:
                continue
            days.append((date, day["price_eur_mwh"].to_numpy(float)))
        if len(days) < 2:
            continue
        hindsight = []
        fto = []
        regrets = []
        used_dates = []
        for idx in range(1, len(days)):
            date, realized = days[idx]
            _, prev_prices = days[idx - 1]
            if len(realized) != len(prev_prices):
                continue
            h_rev = daily_storage_revenue(realized)
            charge_idx = np.argsort(prev_prices)[:4]
            discharge_idx = np.argsort(prev_prices)[-4:]
            f_rev = float(realized[discharge_idx].sum() * 0.92 - realized[charge_idx].sum() / 0.92)
            regret = h_rev - f_rev
            hindsight.append(h_rev)
            fto.append(f_rev)
            regrets.append(regret)
            used_dates.append(date)
            daily_rows.append(
                {
                    "zone": zone,
                    "date": date.isoformat(),
                    "hindsight_revenue": h_rev,
                    "prev_day_fto_revenue": f_rev,
                    "regret": regret,
                }
            )
        rows.append(
            {
                "dataset": "OPSD",
                "zone": zone,
                "days": int(len(regrets)),
                "mean_hindsight_revenue": float(np.mean(hindsight)),
                "mean_prev_day_fto_revenue": float(np.mean(fto)),
                "mean_regret": float(np.mean(regrets)),
                "median_regret": float(np.median(regrets)),
                "negative_revenue_days": int(np.sum(np.asarray(fto) < 0)),
            }
        )
    summary = pd.DataFrame(rows)
    daily = pd.DataFrame(daily_rows)
    summary.to_csv(RESULTS / "opsd_public_vpp_decision_summary.csv", index=False)
    daily.to_csv(RESULTS / "opsd_public_vpp_decision_daily.csv", index=False)
    return summary


def main():
    if not RAW.exists():
        raise SystemExit(f"Missing OPSD raw file: {RAW}")
    tidy = prepare_tidy()
    stats = build_stats(tidy)
    rows = []
    rows.extend(evaluate_target(tidy, "price_eur_mwh", "price_forecasting"))
    rows.extend(evaluate_target(tidy, "load_mw", "load_forecasting"))
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "opsd_public_baselines.csv", index=False)
    decision = evaluate_vpp_decision(tidy)
    print(PROCESSED / "opsd_hourly_price_load_renewables_tidy.csv")
    print(RESULTS / "opsd_public_dataset_stats.csv")
    print(RESULTS / "opsd_public_baselines.csv")
    print(RESULTS / "opsd_public_vpp_decision_summary.csv")
    print(stats.to_string(index=False))
    print(result.sort_values(["task", "zone", "rmse"]).to_string(index=False))
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
