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
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

TARGET = "price_eur_mwh"
ALPHA = 0.10


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


def fit_ridge(frame, feature_cols, target_col=TARGET, l2=1e-5):
    x = frame[feature_cols].to_numpy(float)
    y = frame[target_col].to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    eye = np.eye(x.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + l2 * eye, x.T @ y)


def predict_ridge(frame, feature_cols, beta):
    x = frame[feature_cols].to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    return x @ beta


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


def prepare_frame():
    tidy = pd.read_csv(TIDY, parse_dates=["timestamp_utc"])
    tidy = tidy.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    for lag in [1, 24, 168]:
        tidy[f"price_lag_{lag}"] = tidy.groupby("zone")[TARGET].shift(lag)
    tidy["lag_1_24_abs"] = (tidy["price_lag_1"] - tidy["price_lag_24"]).abs()
    tidy["hour"] = tidy["timestamp_utc"].dt.hour
    tidy["dayofweek"] = tidy["timestamp_utc"].dt.dayofweek
    tidy["month"] = tidy["timestamp_utc"].dt.month
    tidy["hour_sin"] = np.sin(2 * np.pi * tidy["hour"] / 24)
    tidy["hour_cos"] = np.cos(2 * np.pi * tidy["hour"] / 24)
    tidy["dow_sin"] = np.sin(2 * np.pi * tidy["dayofweek"] / 7)
    tidy["dow_cos"] = np.cos(2 * np.pi * tidy["dayofweek"] / 7)
    return tidy


def split_zone(zdf):
    zdf = zdf.sort_values("timestamp_utc").reset_index(drop=True)
    n = len(zdf)
    train_end = int(n * 0.60)
    cal_end = int(n * 0.80)
    return zdf.iloc[:train_end].copy(), zdf.iloc[train_end:cal_end].copy(), zdf.iloc[cal_end:].copy()


def conformal_quantile(abs_resid, alpha=ALPHA):
    abs_resid = np.asarray(abs_resid, dtype=float)
    abs_resid = abs_resid[np.isfinite(abs_resid)]
    if len(abs_resid) == 0:
        return np.nan
    q_level = min(1.0, np.ceil((len(abs_resid) + 1) * (1 - alpha)) / len(abs_resid))
    return float(np.quantile(abs_resid, q_level, method="higher"))


def regime_key(frame, volatility_threshold):
    vol_bin = np.where(frame["lag_1_24_abs"].to_numpy(float) >= volatility_threshold, "highvol", "normal")
    hour_bin = (frame["hour"].to_numpy(int) // 6).astype(str)
    return np.char.add(np.char.add(hour_bin, "_"), vol_bin)


def evaluate_zone(zone, zdf):
    feature_cols = [
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
    clean = zdf[[TARGET, "timestamp_utc", "hour", "lag_1_24_abs"] + feature_cols].dropna().copy()
    train, cal, test = split_zone(clean)
    beta = fit_ridge(train, feature_cols)
    cal_pred = predict_ridge(cal, feature_cols, beta)
    test_pred = predict_ridge(test, feature_cols, beta)
    cal_abs = np.abs(cal[TARGET].to_numpy(float) - cal_pred)
    q_global = conformal_quantile(cal_abs)

    rows = []
    daily = test[["timestamp_utc", TARGET, "lag_1_24_abs", "hour"]].copy()
    daily["pred"] = test_pred

    lower = test_pred - q_global
    upper = test_pred + q_global
    rows.append(
        {
            "dataset": "OPSD",
            "zone": zone,
            "model": "Conformal-linear-global",
            "target_coverage": 1 - ALPHA,
            "calibration_width": 2 * q_global,
            **interval_metrics(test[TARGET], test_pred, lower, upper),
        }
    )
    daily["global_lower"] = lower
    daily["global_upper"] = upper

    volatility_threshold = float(np.nanmedian(cal["lag_1_24_abs"]))
    cal_keys = regime_key(cal, volatility_threshold)
    test_keys = regime_key(test, volatility_threshold)
    q_by_key = {}
    for key in sorted(set(cal_keys)):
        mask = cal_keys == key
        q_by_key[key] = conformal_quantile(cal_abs[mask])
    fallback = q_global
    q_test = np.asarray([q_by_key.get(key, fallback) if np.isfinite(q_by_key.get(key, np.nan)) else fallback for key in test_keys])
    lower_r = test_pred - q_test
    upper_r = test_pred + q_test
    rows.append(
        {
            "dataset": "OPSD",
            "zone": zone,
            "model": "Regime-conformal-linear",
            "target_coverage": 1 - ALPHA,
            "calibration_width": float(np.mean(2 * q_test)),
            **interval_metrics(test[TARGET], test_pred, lower_r, upper_r),
        }
    )
    daily["regime_lower"] = lower_r
    daily["regime_upper"] = upper_r
    daily["regime_width"] = 2 * q_test
    daily["zone"] = zone
    return rows, daily


def run():
    if not TIDY.exists():
        raise SystemExit(f"Missing processed OPSD file: {TIDY}")
    tidy = prepare_frame()
    rows = []
    daily_parts = []
    for zone, zdf in tidy.groupby("zone", sort=True):
        zone_rows, daily = evaluate_zone(zone, zdf)
        rows.extend(zone_rows)
        daily_parts.append(daily)
    result = pd.DataFrame(rows)
    daily = pd.concat(daily_parts, ignore_index=True)
    result.to_csv(RESULTS / "opsd_price_probabilistic_conformal_summary.csv", index=False)
    daily.to_csv(RESULTS / "opsd_price_probabilistic_conformal_daily.csv", index=False)
    return result, daily


def plot_coverage(summary):
    plot = summary.pivot(index="zone", columns="model", values="picp").reset_index()
    zones = plot["zone"].tolist()
    width, height = 1800, 1080
    left, top, right, bottom = 170, 190, 1660, 860
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "OPSD price interval coverage", fill="#172033", font=font(46, True))
    d.text((92, 112), "90% split-conformal target; public day-ahead price test split", fill="#5f6b7a", font=font(25))
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        d.line((left, y, right, y), fill="#D7DCE2", width=1)
        val = i / 5
        d.text((left - 70, y - 13), f"{val:.1f}", fill="#526070", font=font(22))
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    target_y = bottom - 0.90 * (bottom - top)
    d.line((left, target_y, right, target_y), fill="#E45756", width=4)
    bar_w = 95
    group_w = (right - left) / len(zones)
    colors = {"Conformal-linear-global": "#4C78A8", "Regime-conformal-linear": "#F58518"}
    for idx, zone in enumerate(zones):
        base_x = left + idx * group_w + group_w / 2
        for j, model in enumerate(["Conformal-linear-global", "Regime-conformal-linear"]):
            val = float(plot.loc[idx, model])
            x0 = base_x + (j - 0.5) * (bar_w + 18)
            x1 = x0 + bar_w
            y0 = bottom - val * (bottom - top)
            d.rounded_rectangle((x0, y0, x1, bottom), radius=8, fill=colors[model])
            d.text((x0 - 2, y0 - 30), f"{val:.3f}", fill="#1f2937", font=font(21, True))
        tw = d.textlength(zone, font=font(24, True))
        d.text((base_x - tw / 2, bottom + 35), zone, fill="#1f2937", font=font(24, True))
    d.rectangle((290, 930, 324, 958), fill=colors["Conformal-linear-global"])
    d.text((338, 924), "Global conformal", fill="#526070", font=font(24))
    d.rectangle((630, 930, 664, 958), fill=colors["Regime-conformal-linear"])
    d.text((678, 924), "Hour-volatility regime conformal", fill="#526070", font=font(24))
    out = FIGURES / "paper1_fig4_opsd_conformal_coverage.png"
    img.save(out)
    return out


def plot_interval_example(daily):
    sample = daily[daily["zone"] == "DE_LU"].sort_values("timestamp_utc").head(24 * 10).copy()
    width, height = 1800, 1080
    left, top, right, bottom = 165, 200, 1680, 860
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "OPSD conformal price interval example", fill="#172033", font=font(46, True))
    d.text((92, 112), "DE-LU test split, first 10 days; regime-conformal interval", fill="#5f6b7a", font=font(25))
    values = []
    for col in [TARGET, "pred", "regime_lower", "regime_upper"]:
        values.extend(sample[col].to_numpy(float))
    y_min, y_max = float(np.nanmin(values)), float(np.nanmax(values))
    pad = (y_max - y_min) * 0.15 + 1e-6
    y_min -= pad
    y_max += pad
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        d.line((left, y, right, y), fill="#D7DCE2", width=1)
        val = y_min + i * (y_max - y_min) / 5
        d.text((left - 95, y - 13), f"{val:.0f}", fill="#526070", font=font(22))
    d.rectangle((left, top, right, bottom), outline="#8792a2", width=2)

    def pts(vals):
        vals = np.asarray(vals, dtype=float)
        out = []
        for idx, val in enumerate(vals):
            x = left + idx * (right - left) / max(1, len(vals) - 1)
            y = bottom - (val - y_min) / (y_max - y_min) * (bottom - top)
            out.append((x, y))
        return out

    upper = pts(sample["regime_upper"])
    lower = pts(sample["regime_lower"])
    d.polygon(upper + lower[::-1], fill="#D9EAFB")
    d.line(pts(sample[TARGET]), fill="#172033", width=4)
    d.line(pts(sample["pred"]), fill="#E45756", width=4)
    lx, ly = 210, 910
    d.rectangle((lx, ly + 7, lx + 44, ly + 29), fill="#D9EAFB", outline="#A7C7E7")
    d.text((lx + 58, ly), "90% conformal interval", fill="#1f2937", font=font(24))
    lx += 390
    d.line((lx, ly + 18, lx + 58, ly + 18), fill="#172033", width=5)
    d.text((lx + 70, ly), "Realized price", fill="#1f2937", font=font(24))
    lx += 300
    d.line((lx, ly + 18, lx + 58, ly + 18), fill="#E45756", width=5)
    d.text((lx + 70, ly), "Point forecast", fill="#1f2937", font=font(24))
    out = FIGURES / "paper1_fig5_opsd_conformal_interval.png"
    img.save(out)
    return out


def main():
    summary, daily = run()
    coverage_fig = plot_coverage(summary)
    interval_fig = plot_interval_example(daily)
    print(RESULTS / "opsd_price_probabilistic_conformal_summary.csv")
    print(RESULTS / "opsd_price_probabilistic_conformal_daily.csv")
    print(coverage_fig)
    print(interval_fig)
    print(summary.sort_values(["zone", "model"]).to_string(index=False))


if __name__ == "__main__":
    main()
