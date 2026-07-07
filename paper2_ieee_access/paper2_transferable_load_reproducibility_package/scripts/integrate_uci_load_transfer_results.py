from pathlib import Path

import pandas as pd
from docx import Document

from build_paper_package import add_markdown_to_docx, setup_doc


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PKG = ROOT / "manuscript" / "main"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def fmt(x, n=3):
    try:
        if pd.isna(x):
            return ""
        if float(x).is_integer():
            return str(int(float(x)))
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)


def table(df, cols, headers=None):
    headers = headers or cols
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("|" + "|".join(fmt(row[c]) for c in cols) + "|")
    return "\n".join(lines)


def dedupe_lines(text):
    seen = set()
    lines = []
    for line in text.splitlines():
        if line.startswith("- ") and line in seen:
            continue
        if line.startswith("- "):
            seen.add(line)
        lines.append(line)
    return "\n".join(lines) + "\n"


def replace_between(text, start, end, replacement):
    i = text.find(start)
    if i == -1:
        raise ValueError(f"start marker not found: {start}")
    j = text.find(end, i + len(start))
    if j == -1:
        raise ValueError(f"end marker not found after {start}: {end}")
    return text[: i + len(start)] + replacement + text[j:]


def rebuild_docx(md_path):
    text = md_path.read_text(encoding="utf-8")
    doc = Document()
    title = text.splitlines()[0].replace("# ", "").strip()
    setup_doc(doc, title)
    add_markdown_to_docx(doc, text)
    doc.save(md_path.with_suffix(".docx"))


def role_summary(stats):
    rows = []
    for role, rdf in stats.groupby("role", sort=True):
        rows.append(
            {
                "role": role,
                "clients": int(rdf["client"].nunique()),
                "period": "2014-01-01 to 2014-12-31",
                "resolution": "hourly after 15-min aggregation",
                "median_coverage": float(rdf["coverage"].median()),
                "mean_load_min": float(rdf["mean"].min()),
                "mean_load_max": float(rdf["mean"].max()),
            }
        )
    out = pd.DataFrame(rows)
    role_order = {"source": 0, "target": 1}
    out["order"] = out["role"].map(role_order)
    return out.sort_values("order").drop(columns=["order"])


def integrate_paper2():
    path = PKG / "paper_2_transferable_load_forecasting.md"
    text = path.read_text(encoding="utf-8")
    opsd_stats = pd.read_csv(RESULTS / "opsd_public_dataset_stats.csv")
    opsd_baselines = pd.read_csv(RESULTS / "opsd_public_baselines.csv")
    uci_stats = pd.read_csv(RESULTS / "uci_load_dataset_stats.csv")
    uci_summary = pd.read_csv(RESULTS / "uci_load_transfer_summary.csv")
    ssl_summary_path = RESULTS / "uci_ssl_representation_summary.csv"
    ssl_diag_path = RESULTS / "uci_ssl_pretraining_diagnostics.csv"
    cold_summary_path = RESULTS / "uci_ssl_cold_start_summary.csv"
    domain_diag_path = RESULTS / "uci_ssl_domain_shift_diagnostics.csv"
    domain_summary_path = RESULTS / "uci_ssl_domain_shift_summary.csv"
    ssl_summary = pd.read_csv(ssl_summary_path) if ssl_summary_path.exists() else None
    ssl_diag = pd.read_csv(ssl_diag_path) if ssl_diag_path.exists() else None
    cold_summary = pd.read_csv(cold_summary_path) if cold_summary_path.exists() else None
    domain_diag = pd.read_csv(domain_diag_path) if domain_diag_path.exists() else None
    domain_summary = pd.read_csv(domain_summary_path) if domain_summary_path.exists() else None

    opsd_stats["start_date"] = pd.to_datetime(opsd_stats["start_utc"]).dt.strftime("%Y-%m-%d")
    opsd_stats["end_date"] = pd.to_datetime(opsd_stats["end_utc"]).dt.strftime("%Y-%m-%d")
    opsd_load = opsd_baselines[opsd_baselines["task"] == "load_forecasting"].copy()
    opsd_load["model_short"] = opsd_load["model"].replace(
        {
            "Persist-1h": "Persistence",
            "Seasonal-24h": "Seasonal-24h",
            "Seasonal-168h": "Seasonal-168h",
            "Linear-lag-calendar-exog": "Linear lag+cal+exog",
        }
    )
    opsd_best = opsd_load.sort_values(["zone", "rmse"]).groupby("zone", as_index=False).first()
    uci_role = role_summary(uci_stats)

    dataset_section = """

The reproducible public benchmark now has two layers. The first layer uses the OPSD hourly time-series package with load, day-ahead price, wind, and solar variables for DE-LU, DK1, DK2, and Great Britain. It supports within-zone and cross-zone load forecasting at system scale. The second layer uses the UCI Electricity Load Diagrams 2011-2014 dataset as a true multi-client transfer benchmark. The UCI data contain 370 client-level electricity series at 15-minute resolution; the current reproducible run aggregates the selected 2014 subset to hourly resolution and evaluates 30 source clients and 10 target clients.

Table 1 summarizes the public OPSD load benchmark coverage.

""" + table(
        opsd_stats,
        ["zone", "rows", "start_date", "end_date", "load_non_null", "solar_non_null", "wind_non_null"],
        ["Zone", "Rows", "Start", "End", "Load", "Solar", "Wind"],
    ) + """

Table 2 summarizes the UCI multi-client transfer benchmark split used in the transparent public experiment. Clients are selected deterministically by 2014 coverage and load variability, then split into source and target roles.

""" + table(
        uci_role,
        ["role", "clients", "period", "resolution", "median_coverage", "mean_load_min", "mean_load_max"],
        ["Role", "Clients", "Period", "Resolution", "Coverage", "Mean load min", "Mean load max"],
    ) + "\n\n"
    text = replace_between(text, "### 5.1 Datasets", "### 5.2 Baselines", dataset_section)

    ssl_section = ""
    if ssl_summary is not None and ssl_diag is not None:
        selected_ssl = ssl_summary[
            ssl_summary["model"].isin(
                [
                    "SSL-MR-lag+adapter-28d",
                    "SSL-MR-lag-source-head",
                    "SSL-MR-lag+adapter-7d",
                    "SSL-MR+adapter-28d",
                    "SSL-MR+adapter-7d",
                    "SSL-MR-source-head",
                ]
            )
        ].copy()
        selected_ssl = selected_ssl.sort_values("mean_rmse")
        best_base = float(uci_summary.loc[uci_summary["model"] == "Target-28d-linear", "mean_rmse"].iloc[0])
        best_ssl = float(selected_ssl["mean_rmse"].min())
        improvement = (best_base - best_ssl) / best_base * 100
        diag = ssl_diag.iloc[0]
        ssl_section = """

Table 5 reports the first masked-reconstruction representation prototype on the same UCI split. The representation is pretrained on source-client 168-hour history windows using 20% random masking and a 16-dimensional low-rank reconstruction basis. The pretraining run uses 80,000 source windows, explains {explained:.3f} of masked-window variance, and obtains source-window reconstruction RMSE {rec_rmse:.3f} in normalized units.

""".format(
            explained=float(diag["explained_variance_ratio"]),
            rec_rmse=float(diag["source_reconstruction_rmse"]),
        ) + table(
            selected_ssl,
            ["model", "protocol", "target_clients", "mean_mae", "mean_rmse", "mean_smape", "total_n"],
            ["Model", "Protocol", "Targets", "MAE", "RMSE", "sMAPE", "N"],
        ) + """

The best prototype is SSL-MR-lag+adapter-28d, which reduces mean RMSE from 82.784 for the strongest transparent few-shot linear baseline to {best_ssl:.3f}, a relative improvement of {improvement:.2f}%. The diagnostic result is also important: using the masked-reconstruction latent representation alone is weaker than using the latent representation together with lag-1, lag-24, and lag-168 temporal priors. This supports treating self-supervised representation learning and domain-adaptive temporal priors as complementary components rather than as substitutes.
""".format(
            best_ssl=best_ssl,
            improvement=improvement,
        )

    cold_section = ""
    if cold_summary is not None and domain_diag is not None and domain_summary is not None:
        selected_cold = cold_summary[
            cold_summary["model"].isin(
                [
                    "SSL-MR-lag-source-head",
                    "SSL-MR-lag+adapter-1d",
                    "SSL-MR-lag+adapter-3d",
                    "SSL-MR-lag+adapter-7d",
                    "SSL-MR-lag+adapter-28d",
                    "Target-linear-1d",
                    "Target-linear-3d",
                    "Target-linear-7d",
                    "Target-linear-28d",
                    "Seasonal-24h",
                    "Seasonal-168h",
                ]
            )
        ].copy()
        selected_cold = selected_cold.sort_values(["adapt_days", "mean_rmse", "model"])
        selected_cold["model_short"] = selected_cold["model"].replace(
            {
                "SSL-MR-lag-source-head": "SSL source",
                "SSL-MR-lag+adapter-1d": "SSL adapter 1d",
                "SSL-MR-lag+adapter-3d": "SSL adapter 3d",
                "SSL-MR-lag+adapter-7d": "SSL adapter 7d",
                "SSL-MR-lag+adapter-28d": "SSL adapter 28d",
                "Target-linear-1d": "Target ridge 1d",
                "Target-linear-3d": "Target ridge 3d",
                "Target-linear-7d": "Target ridge 7d",
                "Target-linear-28d": "Target ridge 28d",
            }
        )
        selected_cold["protocol_short"] = selected_cold["protocol"].replace(
            {
                "zero-label source representation": "source head",
                "zero-label seasonal": "seasonal",
                "frozen source representation with target adapter": "adapter",
                "target-only lag-calendar ridge": "target ridge",
            }
        )
        domain_table = domain_diag.sort_values("source_target_latent_distance", ascending=False).copy()
        domain_table["distance"] = domain_table["source_target_latent_distance"]
        domain_table["source_rmse"] = domain_table["ssl_source_head_rmse"]
        domain_table["adapter_28d_rmse"] = domain_table["ssl_adapter_28d_rmse"]
        domain_table["gain_pct"] = domain_table["adapter_gain_pct"]
        domain_row = domain_summary.iloc[0]
        cold_section = """

Table 6 extends the UCI experiment to label-scarce and cold-start adaptation. The zero-label row uses the source-trained masked-reconstruction representation and its source head without fitting target labels. The 1-day, 3-day, 7-day, and 28-day rows compare a lightweight target adapter against a target-only lag-calendar ridge model.

""" + table(
            selected_cold,
            ["model_short", "protocol_short", "adapt_days", "target_clients", "mean_mae", "mean_rmse", "mean_smape", "total_n"],
            ["Model", "Protocol", "Days", "Targets", "MAE", "RMSE", "sMAPE", "N"],
        ) + """

The zero-label source representation obtains mean RMSE 75.689, already below the 28-day target-only linear baseline at 82.782. This is the strongest cold-start evidence in the current Paper 2 package. The adapter curve is not monotonic: 1-day and 3-day adapters can overfit or miscalibrate, while 7-day and 28-day adapters return to the 75-RMSE range. The result should be framed carefully: self-supervised source representations reduce the data requirement, but 1-day and 3-day target-label adapters require stronger regularization or meta-validation.

Table 7 reports a representation-domain shift diagnostic. Each target client is represented by the centroid of its 28-day pre-test latent windows. The distance to the source centroid is compared with source-head error and 28-day adapter gain.

""" + table(
            domain_table,
            ["target_client", "distance", "source_rmse", "adapter_28d_rmse", "gain_pct"],
            ["Target", "Latent distance", "Source RMSE", "Adapter RMSE", "Adapter gain %"],
        ) + f"""

The mean source-target latent distance is {float(domain_row["mean_latent_distance"]):.3f}. Across the ten held-out clients, the correlation between latent distance and zero-label source-head RMSE is {float(domain_row["corr_distance_source_rmse"]):.3f}, while the correlation between latent distance and 28-day adapter gain is {float(domain_row["corr_distance_adapter_gain_pct"]):.3f}. In this selected UCI split, latent distance is therefore a useful diagnostic feature but not a sufficient predictor of whether a simple adapter will help.
"""

    results_section = """

Table 3 reports the best transparent OPSD load-forecasting baseline by RMSE for each public zone. The complete baseline file includes one-hour persistence, 24-hour seasonal naive, 168-hour seasonal naive, and lag-calendar-exogenous linear models.

""" + table(
        opsd_best,
        ["zone", "model_short", "mae", "rmse", "smape", "n"],
        ["Zone", "Best baseline", "MAE", "RMSE", "sMAPE", "N"],
    ) + """

The lag-calendar-exogenous linear model is the best transparent baseline on all four public OPSD zones. This result sharpens the claim of the transferable-learning paper: the main contribution is not framed as merely beating weak point-forecasting baselines. Instead, the proposed self-supervised representation is evaluated by cross-domain transfer, few-shot adaptation, and cold-start robustness where reusable temporal representations matter.

Table 4 reports the public UCI multi-client transfer benchmark. The test horizon is October-December 2014 for ten held-out target clients. Source-transfer models train on thirty source clients before the test horizon; few-shot target models use either the seven days or the twenty-eight days immediately before the test horizon. The values are averaged across target clients.

""" + table(
        uci_summary,
        ["model", "protocol", "target_clients", "mean_mae", "mean_rmse", "mean_smape", "total_n"],
        ["Model", "Protocol", "Targets", "MAE", "RMSE", "sMAPE", "N"],
    ) + """

The UCI transfer table is reported as a transparent baseline layer. The best linear-only result is the 28-day target-only linear model, while pooled-source and pooled-plus-target linear transfer models substantially improve over 24-hour and 168-hour seasonal baselines. This pattern sets the reference point for the self-supervised representation experiment: transfer learning helps relative to naive temporal rules, but representation learning must still demonstrate value beyond a well-tuned target-domain linear baseline.

""" + ssl_section + cold_section + """

![Figure 1. Transferable short-term load forecasting framework with source-pooled representation reuse and lightweight adaptation.](./outputs/[run-id]/figures/paper2_fig1_framework.png)

![Figure 2. Local target-domain load curve used as a small adaptation case.](./outputs/[run-id]/figures/paper2_fig2_load_curve.png)

![Figure 3. Pilot RMSE comparison on the local Hunan user-load case.](./outputs/[run-id]/figures/paper2_fig3_load_rmse.png)

![Figure 4. UCI public multi-client transfer benchmark, mean RMSE across ten target clients.](./outputs/[run-id]/figures/paper2_fig4_uci_transfer_rmse.png)

![Figure 5. UCI load-transfer comparison between transparent baselines and the masked-reconstruction representation prototype.](./outputs/[run-id]/figures/paper2_fig5_uci_ssl_prototype_rmse.png)

![Figure 6. UCI label-scarce adaptation curve comparing the source representation, target adapters, and target-only ridge baselines.](./outputs/[run-id]/figures/paper2_fig6_uci_cold_start_curve.png)

![Figure 7. UCI representation-domain shift diagnostic for held-out target clients.](./outputs/[run-id]/figures/paper2_fig7_uci_domain_shift_diagnostic.png)

"""
    text = replace_between(text, "## 6. Results and Discussion", "## 6.2 Pilot Results on Local User Load Data", results_section)
    text = text.replace(
        "The current local user-load curve is small and should not be used as the sole evidence for a transferable-learning paper.",
        "The current local user-load curve is small and is not used as the sole evidence for a transferable-learning paper.",
    )
    text = text.replace(
        "The final paper should therefore use public multi-series load datasets for the main experiment and reserve this local curve for few-shot or local adaptation validation.",
        "The public OPSD and UCI benchmarks form the reproducible main experiment layer, while this local curve is reserved for few-shot or local adaptation validation.",
    )
    text = text.replace(
        "The OPSD public benchmark is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`; additional multi-client transfer-learning data will be documented in the same format.",
        "The OPSD public benchmark is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark is reproducible from `run_uci_load_transfer_baselines.py`.",
    )
    text = text.replace(
        "The OPSD public benchmark is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark is reproducible from `run_uci_load_transfer_baselines.py`.",
        "The OPSD public benchmark is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark is reproducible from `run_uci_load_transfer_baselines.py`, and the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`.",
    )
    text = text.replace(
        "Because the current local user curve is small, the main transferable-learning experiment relies on public multi-series load datasets.",
        "Because the current local user curve is small, the main transferable-learning experiment relies on the public OPSD and UCI multi-series load datasets.",
    )
    text = text.replace(
        "The OPSD public benchmark is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark is reproducible from `run_uci_load_transfer_baselines.py`, and the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`.",
        "The OPSD public benchmark is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark is reproducible from `run_uci_load_transfer_baselines.py`; the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`; and the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`.",
    )
    path.write_text(text, encoding="utf-8")
    rebuild_docx(path)


def update_gap_checklist():
    path = ROOT / "SUBMISSION_GAP_CHECKLIST.md"
    text = path.read_text(encoding="utf-8")
    uci_lines = (
        "- UCI Electricity Load Diagrams has been downloaded, cleaned, sampled into 30 source clients and 10 target clients, and benchmarked for transparent transfer/few-shot baselines.\n"
        "- UCI transfer benchmark table and figure have been integrated into the manuscript results section.\n"
    )
    ssl_lines = (
        "- Initial masked-reconstruction representation prototype with lag priors and target adapter has been implemented on the UCI split.\n"
        "- SSL prototype table and comparison figure have been integrated into the manuscript results section.\n"
        "- UCI label-scarce/cold-start adapter diagnostics now compare zero-label source representation, 1/3/7/28-day adapters, and target-only ridge baselines.\n"
        "- UCI representation-domain shift diagnostics and two additional figures have been integrated into the manuscript results section.\n"
    )
    text = text.replace(uci_lines + uci_lines, uci_lines)
    text = text.replace(ssl_lines + ssl_lines, ssl_lines)
    if uci_lines not in text:
        text = text.replace(
            "- OPSD public load table has been integrated into the manuscript results section.\n",
            "- OPSD public load table has been integrated into the manuscript results section.\n" + uci_lines,
        )
    if ssl_lines not in text:
        text = text.replace(uci_lines, uci_lines + ssl_lines)
    text = text.replace(
        "- Add a true multi-client load benchmark such as UCI Electricity Load Diagrams for transfer learning.\n",
        ""
    )
    text = text.replace(
        "- Run cross-domain/few-shot/cold-start experiments.\n",
        "- Extend the current transparent UCI transfer/few-shot baselines to the final self-supervised pretraining, adapter, and cold-start experiments.\n",
    )
    text = text.replace(
        "- Implement self-supervised pretraining and adapter baselines.\n",
        "",
    )
    text = text.replace(
        "- Extend the current transparent UCI transfer/few-shot baselines to the final self-supervised pretraining, adapter, and cold-start experiments.\n",
        "- Extend the current masked-reconstruction SSL prototype to the final deep self-supervised encoder, adapter ablations, and cold-start experiments.\n",
    )
    text = text.replace(
        "- Extend the current masked-reconstruction SSL prototype to the final deep self-supervised encoder, adapter ablations, and cold-start experiments.\n",
        "- Upgrade the current low-rank masked-reconstruction prototype to a stronger deep encoder and contrastive/pretext objective if the target journal requires a stronger method contribution.\n",
    )
    text = text.replace(
        "- Add representation visualization or domain-shift analysis.\n",
        "- Add reviewer-facing statistical tests and, if time permits, a second public multi-client load dataset.\n",
    )
    text = text.replace(
        "- Replace pilot-only claims with full results.\n",
        "- Convert references to target journal style and confirm target journal classification with school.\n",
    )
    text = dedupe_lines(text)
    path.write_text(text, encoding="utf-8")


def update_runbook():
    path = ROOT / "EXPERIMENT_RUNBOOK.md"
    text = path.read_text(encoding="utf-8")
    uci_line = "- UCI load transfer benchmark: the selected 2014 public run uses 30 source clients, 10 target clients, and 22,080 target test observations; Target-28d-linear is the best transparent baseline with mean RMSE 82.78, while pooled-source transfer reaches mean RMSE 87.32 and improves over seasonal baselines.\n"
    ssl_line = "- UCI SSL prototype: masked reconstruction on 80,000 source windows with 16 components explains 0.817 variance; SSL-MR-lag+adapter-28d reaches mean RMSE 74.96, improving 9.45% over Target-28d-linear on the same target test set.\n"
    cold_line = "- UCI SSL cold-start diagnostics: zero-label SSL-MR-lag-source-head reaches mean RMSE 75.69, below the 28-day target-only linear baseline at 82.78; 1-day and 3-day adapters are unstable, while 7-day and 28-day adapters return to the 75-RMSE range.\n"
    domain_line = "- UCI SSL domain-shift diagnostics: mean source-target latent distance is 1.388; latent distance correlates -0.604 with zero-label source-head RMSE and 0.003 with 28-day adapter gain on the selected ten-client split.\n"
    text = text.replace(uci_line + uci_line, uci_line)
    text = text.replace(ssl_line + ssl_line, ssl_line)
    text = text.replace(cold_line + cold_line, cold_line)
    text = text.replace(domain_line + domain_line, domain_line)
    if uci_line not in text:
        text = text.replace(
            "- Public VPP decision baseline: previous-day forecast-then-optimize storage arbitrage leaves mean regret of 35.08, 28.42, 24.60, and 22.27 EUR/day proxy on DE-LU, DK1, DK2, and Great Britain respectively.\n",
            "- Public VPP decision baseline: previous-day forecast-then-optimize storage arbitrage leaves mean regret of 35.08, 28.42, 24.60, and 22.27 EUR/day proxy on DE-LU, DK1, DK2, and Great Britain respectively.\n"
            + uci_line,
        )
    if ssl_line not in text:
        text = text.replace(uci_line, uci_line + ssl_line)
    if cold_line not in text:
        text = text.replace(ssl_line, ssl_line + cold_line)
    if domain_line not in text:
        text = text.replace(cold_line, cold_line + domain_line)
    text = text.replace(
        "- One public multi-client transfer benchmark still needed: GEFCom or UCI Electricity Load Diagrams.\n",
        "- One public multi-client transfer benchmark: satisfied initially by UCI Electricity Load Diagrams 2011-2014.\n",
    )
    text = text.replace(
        "- Protocols: within-domain, cross-domain, few-shot, cold-start.\n",
        "- Protocols: OPSD within/cross-zone baselines plus UCI source-only transfer and 7-day/28-day few-shot adaptation; cold-start representation experiments remain for the final model.\n",
    )
    text = text.replace(
        "- Protocols: OPSD within/cross-zone baselines plus UCI source-only transfer and 7-day/28-day few-shot adaptation; cold-start representation experiments remain for the final model.\n",
        "- Protocols: OPSD within/cross-zone baselines plus UCI source-only transfer, 7-day/28-day few-shot adaptation, and masked-reconstruction representation adaptation; cold-start experiments remain for the final model.\n",
    )
    text = text.replace(
        "- Protocols: OPSD within/cross-zone baselines plus UCI source-only transfer, 7-day/28-day few-shot adaptation, and masked-reconstruction representation adaptation; cold-start experiments remain for the final model.\n",
        "- Protocols: OPSD within/cross-zone baselines plus UCI source-only transfer, 1/3/7/28-day target adaptation, masked-reconstruction representation adaptation, and latent-domain shift diagnostics.\n",
    )
    text = dedupe_lines(text)
    path.write_text(text, encoding="utf-8")


def main():
    required = [
        RESULTS / "uci_load_dataset_stats.csv",
        RESULTS / "uci_load_transfer_summary.csv",
        RESULTS / "uci_ssl_cold_start_summary.csv",
        RESULTS / "uci_ssl_domain_shift_diagnostics.csv",
        RESULTS / "uci_ssl_domain_shift_summary.csv",
        FIGURES / "paper2_fig4_uci_transfer_rmse.png",
        FIGURES / "paper2_fig6_uci_cold_start_curve.png",
        FIGURES / "paper2_fig7_uci_domain_shift_diagnostic.png",
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(f"missing required UCI output: {path}")
    integrate_paper2()
    update_gap_checklist()
    update_runbook()
    print(PKG / "paper_2_transferable_load_forecasting.md")
    print(PKG / "paper_2_transferable_load_forecasting.docx")
    print(ROOT / "SUBMISSION_GAP_CHECKLIST.md")
    print(ROOT / "EXPERIMENT_RUNBOOK.md")


if __name__ == "__main__":
    main()
