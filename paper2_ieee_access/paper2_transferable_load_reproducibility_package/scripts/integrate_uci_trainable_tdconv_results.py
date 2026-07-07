from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from create_submission_candidate_manuscripts import rebuild_docx


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PKG = ROOT / "manuscript" / "main"
RESULTS = ROOT / "results"
FIG = ROOT / "figures" / "paper2_fig11_uci_trainable_tdconv_baseline.png"

PAPER_FILES = [
    PKG / "paper_2_transferable_load_forecasting.md",
    ROOT / "manuscript" / "submission_candidate" / "paper_2_transferable_load_forecasting.md",
]


MODEL_SHORT = {
    "TDConv-ridge+adapter-28d": "TDConv adapter 28d",
    "TDConv-ridge-source-head": "TDConv source",
    "TDConv-ridge+adapter-7d": "TDConv adapter 7d",
    "TDConv-ridge+target-head-28d": "TDConv target head 28d",
    "RC-lag+adapter-28d": "RC adapter 28d",
    "RC-lag-source-head": "RC source",
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
    tests = pd.read_csv(RESULTS / "uci_trainable_tdconv_client_level_tests.csv")
    summary = pd.read_csv(RESULTS / "uci_trainable_tdconv_baseline_summary.csv").set_index("model")

    td28 = summary.loc["TDConv-ridge+adapter-28d"]
    td0 = summary.loc["TDConv-ridge-source-head"]
    td7 = summary.loc["TDConv-ridge+adapter-7d"]
    target_head = summary.loc["TDConv-ridge+target-head-28d"]

    vs_rc = tests[tests["comparison"] == "TDConv 28d adapter vs RC 28d adapter"].iloc[0]
    vs_rc_source = tests[tests["comparison"] == "TDConv source vs RC source"].iloc[0]
    vs_target = tests[tests["comparison"] == "TDConv 28d adapter vs target ridge 28d"].iloc[0]
    vs_target_head = tests[tests["comparison"] == "TDConv target head 28d vs target ridge 28d"].iloc[0]
    vs_target7 = tests[tests["comparison"] == "TDConv 7d adapter vs target ridge 7d"].iloc[0]

    table = "\n".join(
        [
            "|Check|Baseline|Baseline RMSE|Candidate|Candidate RMSE|Mean gain %|Wins|p sign|",
            "|---|---|---|---|---|---|---|---|",
            comparison_row(tests, "TDConv 28d adapter vs RC 28d adapter", "TDConv 28d vs RC 28d"),
            comparison_row(tests, "TDConv source vs RC source", "TDConv source vs RC source"),
            comparison_row(tests, "TDConv 28d adapter vs target ridge 28d", "TDConv 28d vs target"),
            comparison_row(tests, "TDConv target head 28d vs target ridge 28d", "TDConv target head vs target"),
            comparison_row(tests, "TDConv 7d adapter vs target ridge 7d", "TDConv 7d vs target"),
        ]
    )

    return f"""
Table 10 reports a reviewer-facing trainable encoder check using a dilated-convolution ridge representation. The encoder extracts multi-scale causal slices from each 168-hour history window, fits source-domain ridge heads on standardized convolution-window features, and then applies the same lightweight target-adapter protocol. This check is intentionally described as a trainable dilated-convolution ridge encoder rather than as a full deep TCN, because the current public reproducibility environment uses deterministic NumPy/Pandas training without GPU-dependent deep-learning libraries.

{table}

The trainable dilated-convolution check further strengthens the UCI transfer evidence. The 28-day TDConv adapter obtains mean RMSE {td28.mean_rmse:.3f}, improving over the random-convolution 28-day adapter at {vs_rc.mean_baseline_rmse:.3f} on {int(vs_rc.wins)}/{int(vs_rc.target_clients)} clients with mean paired RMSE gain {vs_rc.mean_rmse_gain_pct:.2f}% (p={vs_rc.sign_test_p_two_sided:.3f}). The zero-label TDConv source head reaches mean RMSE {td0.mean_rmse:.3f}, improving over the random-convolution source head on {int(vs_rc_source.wins)}/{int(vs_rc_source.target_clients)} clients. Relative to the 28-day target-only ridge baseline, the TDConv 28-day adapter wins on {int(vs_target.wins)}/{int(vs_target.target_clients)} clients with mean RMSE gain {vs_target.mean_rmse_gain_pct:.2f}% (p={vs_target.sign_test_p_two_sided:.3f}); the 7-day adapter also wins on {int(vs_target7.wins)}/{int(vs_target7.target_clients)} clients with mean RMSE {td7.mean_rmse:.3f}. The negative control is important: a target-only TDConv head reaches mean RMSE {target_head.mean_rmse:.3f}, worse than the 28-day target ridge by {-vs_target_head.mean_rmse_gain_pct:.2f}% with no sign-test advantage. The result therefore supports source-pooled trainable convolutional representations with regularized adaptation, not unconstrained high-dimensional fitting on small target samples.

![Figure 8. Trainable dilated-convolution ridge encoder check on UCI load transfer.]({FIG})
""".strip()


def update_encoder_note(text: str) -> str:
    old = (
        "In the reproducible public experiment, the encoder family is evaluated in two forms: "
        "a low-rank masked-reconstruction representation and a stronger ROCKET-style random-convolution temporal representation inspired by random convolutional kernel transforms [9], [10]. "
        "The latter is used as an encoder-strengthening check because it tests whether multi-scale temporal filters improve transfer beyond the initial reconstruction basis."
    )
    new = (
        "In the reproducible public experiment, the encoder family is evaluated in three forms: "
        "a low-rank masked-reconstruction representation, a stronger ROCKET-style random-convolution temporal representation inspired by random convolutional kernel transforms [9], [10], and a trainable dilated-convolution ridge encoder. "
        "The latter two checks test whether multi-scale temporal filters and trainable causal convolution-window features improve transfer beyond the initial reconstruction basis."
    )
    if old in text:
        return text.replace(old, new)
    if new in text:
        return text
    marker = "The model is deliberately modular so that alternative encoders can be tested."
    return text.replace(marker, marker + " " + new)


def insert_tdconv_section(text: str, section: str) -> str:
    text = re.sub(
        r"\n\nTable 10 reports a reviewer-facing trainable encoder check.*?!\[Figure \d+\. Trainable dilated-convolution ridge encoder check on UCI load transfer\.\]\(.+?\)",
        "",
        text,
        flags=re.S,
    )

    markers = [
        "\n\nTable 11 adds a nonlinear residual-head check",
        "\n\nTable 12 adds a second public load dataset",
        "\n\nTable 11 adds a second public load dataset",
        "\n\nTable 10 adds a second public load dataset",
    ]
    marker = next((candidate for candidate in markers if candidate in text), "")
    if not marker:
        marker = "\n\n## 6.2 Pilot Results on Local User Load Data"
    if not marker or marker not in text:
        raise ValueError("Paper 2 post-TDConv insertion marker not found")
    text = text.replace(marker, "\n\n" + section + marker, 1)

    text = text.replace("Table 10 adds a second public load dataset", "Table 11 adds a second public load dataset")
    text = text.replace("![Figure 8. Second public load dataset check", "![Figure 9. Second public load dataset check")
    text = text.replace("![Figure 9. Local target-domain load curve", "![Figure 10. Local target-domain load curve")
    text = text.replace("![Figure 10. Pilot RMSE comparison", "![Figure 11. Pilot RMSE comparison")

    text = text.replace(
        "and the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`. The second public load-dataset check",
        "the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`; and the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`. The second public load-dataset check",
    )
    text = text.replace(
        "the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; and the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`.",
        "the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`; and the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`.",
    )
    text = update_encoder_note(text)
    return text


def main() -> None:
    section = make_section()
    for path in PAPER_FILES:
        text = path.read_text(encoding="utf-8")
        text = insert_tdconv_section(text, section)
        path.write_text(text, encoding="utf-8")
        rebuild_docx(path, text)
        print(path)
        print(path.with_suffix(".docx"))


if __name__ == "__main__":
    main()
