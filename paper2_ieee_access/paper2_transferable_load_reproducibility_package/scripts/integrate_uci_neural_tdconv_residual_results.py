from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from create_submission_candidate_manuscripts import rebuild_docx


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PKG = ROOT / "manuscript" / "main"
RESULTS = ROOT / "results"
FIG = ROOT / "figures" / "paper2_fig14_uci_neural_tdconv_residual_check.png"
FIG_REL = "figures/paper2_fig14_uci_neural_tdconv_residual_check.png"

PAPER_FILES = [
    PKG / "paper_2_transferable_load_forecasting.md",
    ROOT / "manuscript" / "submission_candidate" / "paper_2_transferable_load_forecasting.md",
]

MODEL_SHORT = {
    "Neural-TDConv-residual+adapter-28d": "Neural TDConv residual 28d",
    "Neural-TDConv-residual-source-head": "Neural TDConv residual source",
    "Neural-TDConv-residual+adapter-7d": "Neural TDConv residual 7d",
    "TDConv-ridge+adapter-28d": "TDConv adapter 28d",
    "TDConv-ridge-source-head": "TDConv source",
    "RC-lag+adapter-28d": "RC adapter 28d",
    "Target-linear-28d": "Target ridge 28d",
    "Target-linear-7d": "Target ridge 7d",
}


def fmt(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def comparison_row(tests: pd.DataFrame, comparison: str, label: str) -> str:
    row = tests[tests["comparison"] == comparison].iloc[0]
    return (
        "|"
        + "|".join(
            [
                label,
                MODEL_SHORT.get(row.baseline_model, row.baseline_model),
                fmt(row.mean_baseline_rmse),
                MODEL_SHORT.get(row.candidate_model, row.candidate_model),
                fmt(row.mean_candidate_rmse),
                fmt(row.mean_rmse_gain_pct),
                f"{int(row.wins)}/{int(row.target_clients)}",
                fmt(row.sign_test_p_two_sided, 3),
            ]
        )
        + "|"
    )


def make_section() -> str:
    tests = pd.read_csv(RESULTS / "uci_neural_tdconv_residual_client_level_tests.csv")
    summary = pd.read_csv(RESULTS / "uci_neural_tdconv_residual_summary.csv").set_index("model")
    diagnostics = pd.read_csv(RESULTS / "uci_neural_tdconv_residual_training_diagnostics.csv")

    neural28 = summary.loc["Neural-TDConv-residual+adapter-28d"]
    neural0 = summary.loc["Neural-TDConv-residual-source-head"]
    neural7 = summary.loc["Neural-TDConv-residual+adapter-7d"]
    last_diag = diagnostics.iloc[-1]
    best_diag = diagnostics.sort_values("selected_val_rmse").iloc[0]
    vs_td = tests[tests["comparison"] == "Neural residual 28d vs TDConv 28d"].iloc[0]
    vs_td_source = tests[tests["comparison"] == "Neural residual source vs TDConv source"].iloc[0]
    vs_rc = tests[tests["comparison"] == "Neural residual 28d vs RC 28d"].iloc[0]
    vs_target = tests[tests["comparison"] == "Neural residual 28d vs target ridge 28d"].iloc[0]
    vs_target7 = tests[tests["comparison"] == "Neural residual 7d vs target ridge 7d"].iloc[0]

    table = "\n".join(
        [
            "|Check|Baseline|Baseline RMSE|Candidate|Candidate RMSE|Mean gain %|Wins|p sign|",
            "|---|---|---|---|---|---|---|---|",
            comparison_row(tests, "Neural residual 28d vs TDConv 28d", "Neural 28d vs TDConv 28d"),
            comparison_row(tests, "Neural residual source vs TDConv source", "Neural source vs TDConv source"),
            comparison_row(tests, "Neural residual 28d vs RC 28d", "Neural 28d vs RC 28d"),
            comparison_row(tests, "Neural residual 28d vs target ridge 28d", "Neural 28d vs target"),
            comparison_row(tests, "Neural residual 7d vs target ridge 7d", "Neural 7d vs target"),
        ]
    )

    return f"""
Table 11 adds a nonlinear residual-head check on top of the trainable TDConv representation. The residual head is a one-hidden-layer neural model implemented only with NumPy, trained on source-domain residuals after a TDConv ridge head, and selected using source-validation shrinkage rather than target-test feedback. The purpose is reviewer-facing: to test whether a small nonlinear residual correction materially improves the source-pooled representation, or whether the regularized TDConv ridge head already captures the transferable signal.

{table}

The neural residual check is a useful negative control rather than a new headline model. Source-validation selected residual shrinkage {best_diag.selected_residual_shrinkage:.2f}; the selected validation RMSE changes only from {best_diag.base_val_rmse:.3f} for the TDConv ridge base to {best_diag.selected_val_rmse:.3f} after the residual correction. On held-out UCI target clients, the 28-day neural residual adapter reaches mean RMSE {neural28.mean_rmse:.3f}, but it does not improve over the TDConv 28-day adapter ({int(vs_td.wins)}/{int(vs_td.target_clients)} wins, p={vs_td.sign_test_p_two_sided:.3f}). The source-head comparison is similarly neutral ({int(vs_td_source.wins)}/{int(vs_td_source.target_clients)} wins, p={vs_td_source.sign_test_p_two_sided:.3f}). However, the same nonlinear residual model still beats the random-convolution 28-day adapter on {int(vs_rc.wins)}/{int(vs_rc.target_clients)} clients and beats the 28-day target ridge on {int(vs_target.wins)}/{int(vs_target.target_clients)} clients. The result strengthens the methodological boundary of the paper: nonlinear residual capacity is tested, but the evidence favors a parsimonious source-pooled TDConv representation with regularized adaptation rather than a larger neural head fitted for its own sake. The 7-day neural residual adapter remains competitive with mean RMSE {neural7.mean_rmse:.3f}, while the zero-label neural residual source head obtains {neural0.mean_rmse:.3f}; both are reported as robustness evidence rather than as proof of universal neural dominance. The final training epoch residual RMSEs were {last_diag.train_residual_rmse:.3f} on the source-training split and {last_diag.val_residual_rmse:.3f} on the source-validation split.

![Figure 9. Neural TDConv residual-head check on UCI load transfer.]({FIG_REL})
""".strip()


def remove_existing_neural_section(text: str) -> str:
    return re.sub(
        r"\n\nTable 11 adds a nonlinear residual-head check.*?!\[Figure \d+\. Neural TDConv residual-head check on UCI load transfer\.\]\(.+?\)",
        "",
        text,
        flags=re.S,
    )


def renumber_after_insert(text: str) -> str:
    replacements = [
        ("Table 11 adds a second public load dataset", "Table 12 adds a second public load dataset"),
        ("Table 12 reports a multi-horizon robustness extension", "Table 13 reports a multi-horizon robustness extension"),
        ("![Figure 9. Second public load dataset check", "![Figure 10. Second public load dataset check"),
        ("![FIGURE 9. Second public load dataset check", "![FIGURE 10. Second public load dataset check"),
        ("![Figure 10. Multi-horizon robustness check", "![Figure 11. Multi-horizon robustness check"),
        ("![FIGURE 10. Multi-horizon robustness check", "![FIGURE 11. Multi-horizon robustness check"),
        ("![Figure 11. Local target-domain load curve", "![Figure 12. Local target-domain load curve"),
        ("![Figure 11. Pilot RMSE comparison", "![Figure 13. Pilot RMSE comparison"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def update_method_language(text: str) -> str:
    text = text.replace(
        "The strongest UCI evidence comes from source-pooled temporal representations that improve held-out client RMSE against target-only ridge baselines, while the external appliance dataset shows that transparent lag-weather features can remain competitive.",
        "The strongest UCI evidence comes from source-pooled temporal representations that improve held-out client RMSE against target-only ridge baselines; the neural residual check is neutral against the regularized TDConv head but remains stronger than random-convolution and target-only baselines. The external appliance dataset shows that transparent lag-weather features can remain competitive.",
    )
    text = text.replace(
        "masked-reconstruction features, random-convolution features, and trainable dilated-convolution ridge features",
        "masked-reconstruction features, random-convolution features, trainable dilated-convolution ridge features, and a NumPy neural residual-head check",
    )
    text = text.replace(
        "masked-reconstruction, random-convolution, and trainable dilated-convolution representations",
        "masked-reconstruction, random-convolution, trainable dilated-convolution, and neural residual-head checks",
    )
    text = text.replace(
        "Ablations compare masked-reconstruction features, random-convolution features, trainable dilated-convolution ridge features, source heads, target adapters, and target-only negative controls.",
        "Ablations compare masked-reconstruction features, random-convolution features, trainable dilated-convolution ridge features, a neural residual-head extension, source heads, target adapters, and target-only negative controls.",
    )
    text = text.replace(
        "The latter two checks test whether multi-scale temporal filters and trainable causal convolution-window features improve transfer beyond the initial reconstruction basis.",
        "The latter checks test whether multi-scale temporal filters, trainable causal convolution-window features, and a small nonlinear residual correction improve transfer beyond the initial reconstruction basis.",
    )
    text = text.replace(
        "The result therefore supports source-pooled trainable convolutional representations with regularized adaptation, not unconstrained high-dimensional fitting on small target samples.",
        "The result therefore supports source-pooled trainable convolutional representations with regularized adaptation, not unconstrained high-dimensional fitting on small target samples. The next neural residual check tests whether adding nonlinearity changes that conclusion.",
    )
    text = text.replace(
        "representation reuse improves several held-out UCI client protocols and supports label-scarce adaptation, but representation choice, target-label budget, and client mismatch still require validation.",
        "representation reuse improves several held-out UCI client protocols and supports label-scarce adaptation, but representation choice, residual-head capacity, target-label budget, and client mismatch still require validation.",
    )
    text = text.replace(
        "reusable representations improve several held-out UCI client protocols, but representation choice and target-label budget still require validation",
        "reusable representations improve several held-out UCI client protocols, while the neural residual check shows that extra nonlinear capacity does not automatically improve over the regularized TDConv head; representation choice and target-label budget still require validation",
    )
    text = text.replace(
        "reusable representations improve several held-out UCI client protocols, while the neural residual check shows that extra nonlinear capacity does not automatically improve over the regularized TDConv head; representation choice and target-label budget still require validation",
        "representation reuse improves several held-out UCI client protocols and supports label-scarce adaptation, while the neural residual check shows that extra nonlinear capacity does not automatically improve over the regularized TDConv head; representation choice, target-label budget, and failure-aware interpretation still require validation",
    )
    text = text.replace(
        "the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`.",
        "the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`; and the neural TDConv residual-head check is reproducible from `run_uci_neural_tdconv_residual_check.py`.",
    )
    text = text.replace(
        "the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`; and the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`; and the neural TDConv residual-head check is reproducible from `run_uci_neural_tdconv_residual_check.py`.",
        "the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`; the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`; and the neural TDConv residual-head check is reproducible from `run_uci_neural_tdconv_residual_check.py`.",
    )
    return text


def insert_section(text: str, section: str) -> str:
    text = remove_existing_neural_section(text)
    match = re.search(r"\n\nTable \d+ adds a second public load dataset", text)
    marker = match.group(0) if match else "\n\n## 6.2 Pilot Results on Local User Load Data"
    if marker not in text:
        raise ValueError("Paper 2 post-neural insertion marker not found")
    text = text.replace(marker, "\n\n" + section + marker, 1)
    text = renumber_after_insert(text)
    text = update_method_language(text)
    return text


def main() -> None:
    section = make_section()
    for path in PAPER_FILES:
        text = path.read_text(encoding="utf-8")
        text = insert_section(text, section)
        path.write_text(text, encoding="utf-8")
        rebuild_docx(path, text)
        print(path)
        print(path.with_suffix(".docx"))


if __name__ == "__main__":
    main()
