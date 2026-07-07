from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from create_submission_candidate_manuscripts import rebuild_docx


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PKG = ROOT / "manuscript" / "main"
RESULTS = ROOT / "results"
FIG = ROOT / "figures" / "paper2_fig9_uci_random_conv_encoder_comparison.png"

PAPER_FILES = [
    PKG / "paper_2_transferable_load_forecasting.md",
    ROOT / "manuscript" / "submission_candidate" / "paper_2_transferable_load_forecasting.md",
]


def fmt(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


MODEL_SHORT = {
    "RC-lag-source-head": "RC source",
    "RC-lag+adapter-28d": "RC adapter 28d",
    "RC-lag+adapter-7d": "RC adapter 7d",
    "SSL-MR-lag-source-head": "MR source",
    "SSL-MR-lag+adapter-28d": "MR adapter 28d",
    "Target-linear-28d": "Target ridge 28d",
    "Target-linear-7d": "Target ridge 7d",
}


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
    tests = pd.read_csv(RESULTS / "uci_random_conv_client_level_tests.csv")
    summary = pd.read_csv(RESULTS / "uci_random_conv_representation_summary.csv")
    rc28 = summary[summary["model"] == "RC-lag+adapter-28d"].iloc[0]
    rc0 = summary[summary["model"] == "RC-lag-source-head"].iloc[0]
    rc7 = summary[summary["model"] == "RC-lag+adapter-7d"].iloc[0]
    rc3 = summary[summary["model"] == "RC-lag+adapter-3d"].iloc[0]

    table = "\n".join(
        [
            "|Check|Baseline|Baseline RMSE|Candidate|Candidate RMSE|Mean gain %|Wins|p sign|",
            "|---|---|---|---|---|---|---|---|",
            comparison_row(tests, "RC adapter 28d vs MR adapter 28d", "RC 28d vs MR 28d"),
            comparison_row(tests, "RC source vs MR source head", "RC source vs MR source"),
            comparison_row(tests, "RC adapter 28d vs 28d target ridge", "RC 28d vs target"),
            comparison_row(tests, "RC adapter 7d vs 7d target ridge", "RC 7d vs target"),
        ]
    )

    mr28 = tests[tests["comparison"] == "RC adapter 28d vs MR adapter 28d"].iloc[0]
    mr0 = tests[tests["comparison"] == "RC source vs MR source head"].iloc[0]
    target28 = tests[tests["comparison"] == "RC adapter 28d vs 28d target ridge"].iloc[0]
    target7 = tests[tests["comparison"] == "RC adapter 7d vs 7d target ridge"].iloc[0]

    return f"""
Table 9 reports an encoder-strengthening check using a ROCKET-style random-convolution temporal representation. The encoder applies deterministic multi-scale random convolutional filters to the same 168-hour windows, summarizes each filter response by maximum activation, positive-proportion, and terminal activation, and then uses the same source-head and lightweight target-adapter protocol as the masked-reconstruction prototype.

{table}

The random-convolution representation materially strengthens the Paper 2 evidence. The 28-day random-convolution adapter obtains mean RMSE {rc28.mean_rmse:.3f}, compared with {mr28.mean_baseline_rmse:.3f} for the masked-reconstruction 28-day adapter, giving {mr28.mean_rmse_gain_pct:.2f}% mean paired RMSE gain with wins on {int(mr28.wins)}/{int(mr28.target_clients)} clients (p={mr28.sign_test_p_two_sided:.3f}). Even without target labels, the random-convolution source head reaches mean RMSE {rc0.mean_rmse:.3f}, improving over the masked-reconstruction source head on {int(mr0.wins)}/{int(mr0.target_clients)} clients. Relative to target-only ridge baselines, the 28-day random-convolution adapter wins on {int(target28.wins)}/{int(target28.target_clients)} clients with mean gain {target28.mean_rmse_gain_pct:.2f}% (p={target28.sign_test_p_two_sided:.3f}), while the 7-day adapter wins on {int(target7.wins)}/{int(target7.target_clients)} clients with mean gain {target7.mean_rmse_gain_pct:.2f}% (p={target7.sign_test_p_two_sided:.3f}). The 3-day random-convolution adapter remains unstable, with mean RMSE {rc3.mean_rmse:.3f}; therefore, the result supports stronger reusable temporal features, but not unrestricted few-shot adapter fitting without validation.

![Figure 7. Random-convolution representation check on UCI load transfer.]({FIG})
""".strip()


def insert_encoder_note(text: str) -> str:
    note = (
        " In the reproducible public experiment, the encoder family is evaluated in two forms: "
        "a low-rank masked-reconstruction representation and a stronger ROCKET-style random-convolution "
        "temporal representation. The latter is used as an encoder-strengthening check because it tests "
        "whether multi-scale temporal filters improve transfer beyond the initial reconstruction basis."
    )
    target = "The model is deliberately modular so that alternative encoders can be tested."
    replacement = target + note
    if note in text:
        return text
    return text.replace(target, replacement)


def insert_section(text: str, section: str) -> str:
    text = re.sub(
        r"\n\nTable 9 reports an encoder-strengthening check.*?(?=\n\n## 6\.2 Pilot Results on Local User Load Data)",
        "",
        text,
        flags=re.S,
    )
    marker = "\n\n## 6.2 Pilot Results on Local User Load Data"
    if marker not in text:
        raise ValueError("Paper 2 local pilot section marker not found")
    text = text.replace(marker, "\n\n" + section + marker)
    text = text.replace(
        "and the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`.",
        "the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; and the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`.",
    )
    text = text.replace(
        "the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; and the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`.",
        "the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; and the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`.",
    )
    return insert_encoder_note(text)


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
