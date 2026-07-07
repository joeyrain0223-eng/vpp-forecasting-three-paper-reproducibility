from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from create_submission_candidate_manuscripts import rebuild_docx


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PKG = ROOT / "manuscript" / "main"
RESULTS = ROOT / "results"
FIG = ROOT / "figures" / "paper2_fig10_uci_appliances_second_dataset.png"
FIG_MULTI = ROOT / "figures" / "paper2_fig13_uci_appliances_multihorizon_robustness.png"
FIG_LINK = "figures/paper2_fig10_uci_appliances_second_dataset.png"
FIG_MULTI_LINK = "figures/paper2_fig13_uci_appliances_multihorizon_robustness.png"

TARGETS = [
    PKG / "paper_2_transferable_load_forecasting.md",
    ROOT / "manuscript" / "submission_candidate" / "paper_2_transferable_load_forecasting.md",
]


def markdown_table(df: pd.DataFrame) -> str:
    rows = [["Model", "Protocol", "MAE (Wh)", "RMSE (Wh)", "sMAPE (%)", "Test rows"]]
    for row in df.itertuples(index=False):
        rows.append(
            [
                row.model,
                row.protocol,
                f"{float(row.mae_wh):.2f}",
                f"{float(row.rmse_wh):.2f}",
                f"{float(row.smape):.2f}",
                str(int(row.n)),
            ]
        )
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    out.append("|" + "|".join(str(c).ljust(widths[i]) for i, c in enumerate(rows[0])) + "|")
    out.append("|" + "|".join("-" * widths[i] for i in range(len(rows[0]))) + "|")
    for r in rows[1:]:
        out.append("|" + "|".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + "|")
    return "\n".join(out)


def build_section() -> str:
    results = pd.read_csv(RESULTS / "uci_appliances_energy_baselines.csv").sort_values("rmse_wh")
    stats = pd.read_csv(RESULTS / "uci_appliances_energy_dataset_stats.csv").iloc[0]
    mh = pd.read_csv(RESULTS / "uci_appliances_energy_multihorizon_summary.csv")
    mh["order"] = mh["horizon"].map({"1h": 1, "3h": 3, "6h": 6, "12h": 12})
    mh = mh.sort_values("order")
    best = results.iloc[0]
    persistence = results[results["model"] == "Persistence-current"].iloc[0]
    seasonal = results[results["model"] == "Seasonal-24h"].iloc[0]
    random_row = results[results["model"] == "Random-window ridge"].iloc[0]
    gain_vs_persistence = (float(persistence["rmse_wh"]) - float(best["rmse_wh"])) / float(persistence["rmse_wh"]) * 100
    gain_vs_seasonal = (float(seasonal["rmse_wh"]) - float(best["rmse_wh"])) / float(seasonal["rmse_wh"]) * 100
    random_gap = (float(random_row["rmse_wh"]) - float(best["rmse_wh"])) / float(best["rmse_wh"]) * 100
    mh_rows = [["Horizon", "Best model", "Best RMSE", "Lag-weather RMSE", "Random-window RMSE", "Gain vs persistence"]]
    for row in mh.itertuples(index=False):
        mh_rows.append(
            [
                row.horizon,
                row.best_model,
                f"{float(row.best_rmse_wh):.2f}",
                f"{float(row.lag_weather_rmse_wh):.2f}",
                f"{float(row.random_window_rmse_wh):.2f}",
                f"{float(row.lag_weather_gain_vs_persistence_pct):.2f}%",
            ]
        )
    widths = [max(len(str(r[i])) for r in mh_rows) for i in range(len(mh_rows[0]))]
    mh_table = []
    mh_table.append("|" + "|".join(str(c).ljust(widths[i]) for i, c in enumerate(mh_rows[0])) + "|")
    mh_table.append("|" + "|".join("-" * widths[i] for i in range(len(mh_rows[0]))) + "|")
    for r in mh_rows[1:]:
        mh_table.append("|" + "|".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + "|")
    mh_table = "\n".join(mh_table)
    random_best = mh[mh["best_model"] == "Random-window ridge"]
    horizons_random_best = ", ".join(random_best["horizon"].tolist())
    random_best_min_gain = float(random_best["random_window_gain_vs_lag_weather_pct"].min())
    random_best_max_gain = float(random_best["random_window_gain_vs_lag_weather_pct"].max())
    return f"""
Table 11 adds a second public load dataset, UCI Appliances Energy Prediction [21], as an external sanity check beyond the UCI Electricity Load Diagrams multi-client split. The data contain 10-minute appliance-energy observations and environmental covariates from a residential setting; this paper converts them into a one-hour-ahead chronological forecasting task with {int(stats['train_rows'])} training rows, {int(stats['validation_rows'])} validation rows, and {int(stats['test_rows'])} final test rows.

{markdown_table(results)}

The second-dataset result supports the paper's conservative claim. A lag-weather ridge model reaches {float(best['rmse_wh']):.2f} Wh RMSE, improving over current-value persistence by {gain_vs_persistence:.2f}% and over the 24-hour seasonal baseline by {gain_vs_seasonal:.2f}%. The deterministic random-window representation is essentially tied with the lag-weather ridge model, with only {random_gap:.2f}% higher RMSE. This means the external dataset validates the importance of lag, weather, and calendar features, but it does not justify claiming that random temporal representations universally dominate a well-specified transparent model.

![Figure 9. Second public load dataset check using UCI Appliances Energy Prediction.]({FIG_LINK})

Table 12 reports a multi-horizon robustness extension on the same public UCI Appliances holdout. The task is repeated at 1-hour, 3-hour, 6-hour, and 12-hour horizons using only current and historical information. The lag-weather ridge remains the best 1-hour model, while the deterministic random-window ridge is best at {horizons_random_best}; its advantage over lag-weather ridge is small on those longer horizons, ranging from {random_best_min_gain:.2f}% to {random_best_max_gain:.2f}% relative RMSE improvement. This result strengthens the external robustness evidence while keeping the claim bounded: random temporal filters can help slightly at longer horizons, but the paper's main contribution remains source-pooled representation reuse under cross-client transfer, not universal dominance on every public load task.

{mh_table}

![Figure 10. Multi-horizon robustness check on UCI Appliances Energy Prediction.]({FIG_MULTI_LINK})
""".strip()


def update_text(text: str) -> str:
    section = build_section()
    marker_match = re.search(r"Table (\d+) adds a second public load dataset", text)
    if marker_match:
        table_number = int(marker_match.group(1))
        figure_number = table_number if table_number >= 12 else 9
        section = section.replace("Table 11 adds a second public load dataset", f"Table {table_number} adds a second public load dataset")
        section = section.replace("Table 12 reports a multi-horizon robustness extension", f"Table {table_number + 1} reports a multi-horizon robustness extension")
        section = section.replace("Figure 9. Second public load dataset check", f"Figure {figure_number}. Second public load dataset check")
        section = section.replace("Figure 10. Multi-horizon robustness check", f"Figure {figure_number + 1}. Multi-horizon robustness check")
        start = marker_match.start()
        anchors = [
            "## 6.2 Pilot Results on Local User Load Data",
            "## 7. Conclusion",
        ]
        end_candidates = [text.index(anchor, start) for anchor in anchors if anchor in text[start:]]
        if not end_candidates:
            raise ValueError("Paper 2 post-UCI-Appliances anchor not found")
        end = min(end_candidates)
        text = text[:start] + section + "\n\n" + text[end:]
    else:
        anchor = "## 6.2 Pilot Results on Local User Load Data"
        if anchor not in text:
            raise ValueError("Paper 2 local pilot section anchor not found")
        text = text.replace(anchor, section + "\n\n" + anchor, 1)

    text = text.replace(
        "The reproducible public benchmark now has two layers. The first layer uses the OPSD hourly time-series package with load, day-ahead price, wind, and solar variables for DE-LU, DK1, DK2, and Great Britain [18]. It supports within-zone and cross-zone load forecasting at system scale. The second layer uses the UCI Electricity Load Diagrams 2011-2014 dataset as a true multi-client transfer benchmark [17]. The UCI data contain 370 client-level electricity series at 15-minute resolution; the current reproducible run aggregates the selected 2014 subset to hourly resolution and evaluates 30 source clients and 10 target clients.",
        "The reproducible public benchmark now has three layers. The first layer uses the OPSD hourly time-series package with load, day-ahead price, wind, and solar variables for DE-LU, DK1, DK2, and Great Britain [18]. It supports within-zone and cross-zone load forecasting at system scale. The second layer uses the UCI Electricity Load Diagrams 2011-2014 dataset as a true multi-client transfer benchmark [17]. The UCI data contain 370 client-level electricity series at 15-minute resolution; the current reproducible run aggregates the selected 2014 subset to hourly resolution and evaluates 30 source clients and 10 target clients. The third layer uses UCI Appliances Energy Prediction [21] as an external single-site public load dataset with weather and calendar covariates, used to test whether the feature and representation claims survive outside the multi-client UCI split.",
    )
    text = text.replace(
        "The OPSD public benchmark [18] is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark [17] is reproducible from `run_uci_load_transfer_baselines.py`; the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`; the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; and the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`. Main claims should be reproducible without relying solely on private data.",
        "The OPSD public benchmark [18] is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark [17] is reproducible from `run_uci_load_transfer_baselines.py`; the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`; the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; and the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`. The second public load-dataset and multi-horizon checks on UCI Appliances Energy Prediction [21] are reproducible from `run_uci_appliances_energy_baselines.py` and `run_uci_appliances_multihorizon_robustness.py`. Main claims should be reproducible without relying solely on private data.",
    )
    text = text.replace(
        "The OPSD public benchmark [18] is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark [17] is reproducible from `run_uci_load_transfer_baselines.py`; the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`; the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; and the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`. Main claims are reproducible without relying solely on private data.",
        "The OPSD public benchmark [18] is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark [17] is reproducible from `run_uci_load_transfer_baselines.py`; the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`; the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; and the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`. The second public load-dataset and multi-horizon checks on UCI Appliances Energy Prediction [21] are reproducible from `run_uci_appliances_energy_baselines.py` and `run_uci_appliances_multihorizon_robustness.py`. Main claims are reproducible without relying solely on private data.",
    )
    text = text.replace(
        "The OPSD public benchmark [18] is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark [17] is reproducible from `run_uci_load_transfer_baselines.py`; the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`; the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`; and the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`. The second public load-dataset check on UCI Appliances Energy Prediction [21] is reproducible from `run_uci_appliances_energy_baselines.py`. Main claims are reproducible without relying solely on private data.",
        "The OPSD public benchmark [18] is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark [17] is reproducible from `run_uci_load_transfer_baselines.py`; the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`; the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`; and the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`. The second public load-dataset and multi-horizon checks on UCI Appliances Energy Prediction [21] are reproducible from `run_uci_appliances_energy_baselines.py` and `run_uci_appliances_multihorizon_robustness.py`. Main claims are reproducible without relying solely on private data.",
    )
    text = text.replace(
        "transparent lag-weather models remain competitive on the external UCI Appliances check.",
        "transparent lag-weather models remain competitive on the external UCI Appliances one-hour and multi-horizon checks.",
    )
    text = text.replace(
        "The external UCI Appliances check further motivates a failure-aware interpretation, because transparent lag-weather models remain competitive when the public task is dominated by ordinary lag, calendar, and environmental structure.",
        "The external UCI Appliances one-hour and multi-horizon checks further motivate a failure-aware interpretation, because transparent lag-weather models remain competitive and random temporal filters only provide small longer-horizon gains when the public task is dominated by ordinary lag, calendar, and environmental structure.",
    )
    if "[21] Candanedo" not in text:
        text = text.replace(
            "Note: final manuscript submission should replace this compact seed list with the exact target journal reference style and add any additional domain-specific references required by reviewers.",
            '[21] L. M. Candanedo, V. Feldheim, and D. Deramaix, "Appliances Energy Prediction," UCI Machine Learning Repository, 2017. https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction.\n\nNote: final manuscript submission should replace this compact seed list with the exact target journal reference style and add any additional domain-specific references required by reviewers.',
        )
    text = text.replace(
        "![Figure 10. Local target-domain load curve used as a small adaptation case.]",
        "![Figure 11. Local target-domain load curve used as a small adaptation case.]",
    )
    return text


def main() -> None:
    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        updated = update_text(text)
        path.write_text(updated, encoding="utf-8")
        rebuild_docx(path, updated)
        print(path)
        print(path.with_suffix(".docx"))


if __name__ == "__main__":
    main()
