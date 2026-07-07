from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from create_submission_candidate_manuscripts import rebuild_docx


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PKG = ROOT / "manuscript" / "main"
RESULTS = ROOT / "results"
FIG = ROOT / "figures" / "paper2_fig8_uci_client_level_stat_tests.png"

PAPER_FILES = [
    PKG / "paper_2_transferable_load_forecasting.md",
    ROOT / "manuscript" / "submission_candidate" / "paper_2_transferable_load_forecasting.md",
]


def fmt(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def load_table() -> pd.DataFrame:
    tests = pd.read_csv(RESULTS / "uci_ssl_client_level_stat_tests.csv")
    order = [
        "zero-label SSL source head vs 28d target ridge",
        "28d SSL adapter vs 28d target ridge",
        "7d SSL adapter vs 7d target ridge",
        "1d SSL adapter vs 1d target ridge",
        "zero-label SSL source head vs seasonal-168h",
        "zero-label SSL source head vs seasonal-24h",
    ]
    tests["order"] = tests["comparison"].apply(order.index)
    return tests.sort_values("order")


def make_section() -> str:
    tests = load_table()
    rows = []
    short = {
        "zero-label SSL source head vs 28d target ridge": "SSL source vs target ridge 28d",
        "28d SSL adapter vs 28d target ridge": "SSL adapter 28d vs target ridge 28d",
        "7d SSL adapter vs 7d target ridge": "SSL adapter 7d vs target ridge 7d",
        "1d SSL adapter vs 1d target ridge": "SSL adapter 1d vs target ridge 1d",
        "zero-label SSL source head vs seasonal-168h": "SSL source vs seasonal 168h",
        "zero-label SSL source head vs seasonal-24h": "SSL source vs seasonal 24h",
    }
    for row in tests.itertuples(index=False):
        rows.append(
            "|"
            + "|".join(
                [
                    short[row.comparison],
                    fmt(row.mean_baseline_rmse),
                    fmt(row.mean_candidate_rmse),
                    fmt(row.mean_rmse_improvement_pct),
                    f"{int(row.wins)}/{int(row.target_clients)}",
                    fmt(row.sign_test_p_two_sided, 3),
                ]
            )
            + "|"
        )
    table = "\n".join(
        [
            "|Comparison|Baseline RMSE|SSL RMSE|Mean gain %|Wins|p sign|",
            "|---|---|---|---|---|---|",
            *rows,
        ]
    )
    source = tests[tests["comparison"] == "zero-label SSL source head vs 28d target ridge"].iloc[0]
    adapter28 = tests[tests["comparison"] == "28d SSL adapter vs 28d target ridge"].iloc[0]
    adapter7 = tests[tests["comparison"] == "7d SSL adapter vs 7d target ridge"].iloc[0]
    adapter1 = tests[tests["comparison"] == "1d SSL adapter vs 1d target ridge"].iloc[0]
    seasonal168 = tests[tests["comparison"] == "zero-label SSL source head vs seasonal-168h"].iloc[0]
    return f"""
Table 8 reports paired client-level evidence on the same ten UCI target clients. The paired unit is the target client, and the exact two-sided sign test evaluates whether the SSL representation has lower RMSE than the matched baseline on more clients than expected by chance.

{table}

The client-level test strengthens the Paper 2 claim. The zero-label source representation beats the strongest 28-day target-only ridge baseline on {int(source.wins)}/{int(source.target_clients)} clients, with mean RMSE gain {source.mean_rmse_improvement_pct:.2f}% and exact sign-test p={source.sign_test_p_two_sided:.3f}. The 28-day SSL adapter also wins on {int(adapter28.wins)}/{int(adapter28.target_clients)} clients with mean gain {adapter28.mean_rmse_improvement_pct:.2f}% (p={adapter28.sign_test_p_two_sided:.3f}), while the 7-day adapter wins on {int(adapter7.wins)}/{int(adapter7.target_clients)} clients against the 7-day target ridge with mean gain {adapter7.mean_rmse_improvement_pct:.2f}% (p={adapter7.sign_test_p_two_sided:.3f}). The 1-day adapter beats the 1-day target ridge on all clients, but this should be interpreted as low-label robustness relative to a weak same-budget target baseline rather than as the best overall model. Against seasonal rules, the zero-label source representation wins on all clients versus both weekly and daily seasonal baselines, with mean gains {seasonal168.mean_rmse_improvement_pct:.2f}% and {tests[tests["comparison"] == "zero-label SSL source head vs seasonal-24h"].iloc[0].mean_rmse_improvement_pct:.2f}%. These p-values are reported as unadjusted descriptive paired sign-test values; interpretation relies on repeated direction, effect size, and negative-control behavior, not as a claim that all encoder variants have been exhaustively optimized. The negative client case in Figure 6, MT_163, is retained as limitation evidence: source representations reduce average and paired error, but a portable adapter still needs target-domain model selection.

![Figure 6. UCI client-level paired RMSE improvements for self-supervised representation models.]({FIG})
""".strip()


def insert_section(text: str, section: str) -> str:
    text = re.sub(
        r"\n\nTable 8 reports paired client-level evidence.*?(?=\n\n## 6\.2 Pilot Results on Local User Load Data)",
        "",
        text,
        flags=re.S,
    )
    marker = "\n\n## 6.2 Pilot Results on Local User Load Data"
    if marker not in text:
        raise ValueError("Paper 2 local pilot section marker not found")
    text = text.replace(marker, "\n\n" + section + marker)
    text = text.replace(
        "and the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`.",
        "the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; and the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`.",
    )
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
