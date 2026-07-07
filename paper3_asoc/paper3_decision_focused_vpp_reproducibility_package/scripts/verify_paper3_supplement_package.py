from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
SUPP_ROOT = ROOT.parent
SUPP = ROOT
ZIP_PATH = ROOT.parent / "paper3_decision_focused_vpp_reproducibility_package.zip"
AUDIT = ROOT.parent / "paper3_decision_focused_vpp_reproducibility_package_audit.md"

FORBIDDEN_MANUSCRIPT_CLAIMS = [
    "embeds a differentiable or surrogate "
    "optimization layer into the learning process",
    "gradients are propagated from decision loss "
    "to forecasting parameters",
    "This paper proposes a decision-focused learning "
    "framework for virtual power plant bidding.",
]

REQUIRED_MANUSCRIPT_CLAIMS = [
    "auditable forecast-to-decision",
    "policy-search",
    "settlement",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest() -> tuple[list[dict], list[str], list[str]]:
    rows = list(csv.DictReader((SUPP / "MANIFEST.csv").open(encoding="utf-8")))
    missing = []
    mismatched = []
    for row in rows:
        path = SUPP / row["relative_path"]
        if not path.exists():
            missing.append(row["relative_path"])
            continue
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            mismatched.append(row["relative_path"])
    return rows, missing, mismatched


def verify_zip() -> tuple[int, int, int, str]:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        bad = zf.testzip()
        infos = zf.infolist()
        return len(infos), sum(i.file_size for i in infos), sum(i.compress_size for i in infos), bad or ""


def require_close(actual: float, expected: float, tol: float = 0.01) -> bool:
    return abs(float(actual) - expected) <= tol


def verify_key_results() -> dict:
    result_dir = SUPP / "results"
    risk = pd.read_csv(result_dir / "opsd_vpp_risk_extended_summary.csv")
    risk_avg = (
        risk.groupby("method", as_index=False)
        .agg(
            mean_revenue=("mean_revenue", "mean"),
            cvar_10=("cvar_10", "mean"),
            negative_revenue_days=("negative_revenue_days", "sum"),
        )
        .set_index("method")
    )
    policy = pd.read_csv(result_dir / "opsd_decision_focused_policy_test_aggregate.csv").set_index("method")
    coupled = pd.read_csv(result_dir / "opsd_forecast_coupled_vpp_test_aggregate.csv").set_index("method")
    risk_sweep = pd.read_csv(result_dir / "opsd_risk_aversion_sensitivity_test_aggregate.csv").set_index("risk_weight")
    stress = pd.read_csv(result_dir / "opsd_vpp_reviewer_robustness_stress_aggregate.csv")
    high = stress[(stress["scenario"] == "High penalty + transaction cost")].set_index("method")
    fine = pd.read_csv(result_dir / "opsd_vpp_reviewer_robustness_fine_grid_summary.csv").set_index("method")
    q_learning = pd.read_csv(result_dir / "opsd_constrained_q_learning_vpp_test_aggregate.csv").set_index("method")
    q_paired = pd.read_csv(result_dir / "opsd_constrained_q_learning_vpp_paired_tests.csv")
    genetic = pd.read_csv(result_dir / "opsd_genetic_policy_search_vpp_test_aggregate.csv").set_index("method")
    genetic_paired = pd.read_csv(result_dir / "opsd_genetic_policy_search_vpp_paired_tests.csv")
    genetic_multiseed = pd.read_csv(result_dir / "opsd_genetic_policy_multiseed_stability_summary.csv").set_index("metric")
    pso = pd.read_csv(result_dir / "opsd_pso_policy_search_vpp_test_aggregate.csv").set_index("method")
    pso_paired = pd.read_csv(result_dir / "opsd_pso_policy_search_vpp_paired_tests.csv")
    surrogate = pd.read_csv(result_dir / "opsd_surrogate_policy_gradient_vpp_test_aggregate.csv").set_index("method")
    surrogate_paired = pd.read_csv(result_dir / "opsd_surrogate_policy_gradient_vpp_paired_tests.csv")
    reg = pd.read_csv(SUPP / "references" / "paper3_verified_reference_register.csv")

    high_best = high["mean_revenue"].idxmax()
    fine_best_revenue = fine["mean_revenue"].idxmax()
    fine_best_cvar = fine["cvar_10"].idxmax()

    checks = {
        "risk_rolling_revenue": float(risk_avg.loc["Rolling-28d mean FTO", "mean_revenue"]),
        "risk_robust_revenue": float(risk_avg.loc["Robust quantile FTO", "mean_revenue"]),
        "risk_prev_revenue": float(risk_avg.loc["Prev-day FTO", "mean_revenue"]),
        "policy_df_revenue": float(policy.loc["DF policy search (revenue)", "mean_revenue"]),
        "coupled_df_revenue": float(coupled.loc["Forecast-coupled DF (revenue)", "mean_revenue"]),
        "coupled_df_regret": float(coupled.loc["Forecast-coupled DF (revenue)", "mean_regret"]),
        "risk_sweep_lambda0_cvar": float(risk_sweep.loc[0.0, "cvar_10"]),
        "risk_sweep_lambda025_cvar": float(risk_sweep.loc[0.25, "cvar_10"]),
        "high_stress_best_revenue_method": high_best,
        "fine_best_revenue_method": fine_best_revenue,
        "fine_best_cvar_method": fine_best_cvar,
        "q_learning_revenue": float(q_learning.loc["Constrained Q-learning policy", "mean_revenue"]),
        "q_learning_cvar": float(q_learning.loc["Constrained Q-learning policy", "cvar_10"]),
        "q_learning_vs_df_losses": int(
            q_paired[
                q_paired["comparison"]
                == "Constrained Q-learning policy vs DF policy search (revenue)"
            ]["losses"].iloc[0]
        ),
        "genetic_risk_revenue": float(genetic.loc["Genetic policy search (risk-adjusted)", "mean_revenue"]),
        "genetic_risk_cvar": float(genetic.loc["Genetic policy search (risk-adjusted)", "cvar_10"]),
        "genetic_risk_loss_days": int(genetic.loc["Genetic policy search (risk-adjusted)", "negative_revenue_days"]),
        "genetic_vs_df_revenue_wins": int(
            genetic_paired[
                genetic_paired["comparison"]
                == "Genetic policy search (risk-adjusted) vs DF policy search (revenue)"
            ]["wins"].iloc[0]
        ),
        "genetic_vs_df_revenue_losses": int(
            genetic_paired[
                genetic_paired["comparison"]
                == "Genetic policy search (risk-adjusted) vs DF policy search (revenue)"
            ]["losses"].iloc[0]
        ),
        "genetic_multiseed_revenue_mean": float(genetic_multiseed.loc["mean_revenue_mean", "value"]),
        "genetic_multiseed_revenue_std": float(genetic_multiseed.loc["mean_revenue_std", "value"]),
        "genetic_multiseed_cvar_mean": float(genetic_multiseed.loc["cvar_10_mean", "value"]),
        "genetic_multiseed_cvar_std": float(genetic_multiseed.loc["cvar_10_std", "value"]),
        "genetic_multiseed_seeds_above_df": int(genetic_multiseed.loc["seeds_at_or_above_df_revenue", "value"]),
        "pso_risk_revenue": float(pso.loc["PSO policy search (risk-adjusted)", "mean_revenue"]),
        "pso_risk_cvar": float(pso.loc["PSO policy search (risk-adjusted)", "cvar_10"]),
        "pso_risk_loss_days": int(pso.loc["PSO policy search (risk-adjusted)", "negative_revenue_days"]),
        "pso_vs_genetic_risk_wins": int(
            pso_paired[
                pso_paired["comparison"]
                == "PSO policy search (risk-adjusted) vs Genetic policy search (risk-adjusted)"
            ]["wins"].iloc[0]
        ),
        "pso_vs_genetic_risk_losses": int(
            pso_paired[
                pso_paired["comparison"]
                == "PSO policy search (risk-adjusted) vs Genetic policy search (risk-adjusted)"
            ]["losses"].iloc[0]
        ),
        "surrogate_risk_revenue": float(surrogate.loc["Surrogate policy-gradient (risk-adjusted)", "mean_revenue"]),
        "surrogate_risk_cvar": float(surrogate.loc["Surrogate policy-gradient (risk-adjusted)", "cvar_10"]),
        "surrogate_risk_loss_days": int(surrogate.loc["Surrogate policy-gradient (risk-adjusted)", "negative_revenue_days"]),
        "surrogate_vs_genetic_risk_wins": int(
            surrogate_paired[
                surrogate_paired["comparison"]
                == "Surrogate policy-gradient (risk-adjusted) vs Genetic policy search (risk-adjusted)"
            ]["wins"].iloc[0]
        ),
        "surrogate_vs_genetic_risk_losses": int(
            surrogate_paired[
                surrogate_paired["comparison"]
                == "Surrogate policy-gradient (risk-adjusted) vs Genetic policy search (risk-adjusted)"
            ]["losses"].iloc[0]
        ),
        "surrogate_vs_q_learning_wins": int(
            surrogate_paired[
                surrogate_paired["comparison"]
                == "Surrogate policy-gradient (risk-adjusted) vs Constrained Q-learning policy"
            ]["wins"].iloc[0]
        ),
        "surrogate_vs_q_learning_losses": int(
            surrogate_paired[
                surrogate_paired["comparison"]
                == "Surrogate policy-gradient (risk-adjusted) vs Constrained Q-learning policy"
            ]["losses"].iloc[0]
        ),
        "doi_verified": int((reg["status"] == "doi_verified").sum()),
        "url_verified": int((reg["status"] == "url_verified_no_doi").sum()),
        "review_required": int((~reg["status"].isin(["doi_verified", "url_verified_no_doi"])).sum()),
    }
    checks["numeric_checks_pass"] = all(
        [
            require_close(checks["risk_rolling_revenue"], 20.881, 0.02),
            require_close(checks["risk_robust_revenue"], 20.101, 0.02),
            require_close(checks["risk_prev_revenue"], 17.389, 0.02),
            require_close(checks["policy_df_revenue"], 21.639, 0.02),
            require_close(checks["coupled_df_revenue"], 31.435, 0.02),
            require_close(checks["coupled_df_regret"], 4.669, 0.02),
            require_close(checks["risk_sweep_lambda0_cvar"], -1.853, 0.02),
            require_close(checks["risk_sweep_lambda025_cvar"], -1.276, 0.02),
            checks["high_stress_best_revenue_method"] == "DF policy search (revenue)",
            checks["fine_best_revenue_method"] == "Coarse revenue grid",
            checks["fine_best_cvar_method"] == "Coarse risk-adjusted grid",
            require_close(checks["q_learning_revenue"], -5.655, 0.02),
            require_close(checks["q_learning_cvar"], -24.534, 0.03),
            checks["q_learning_vs_df_losses"] == 1280,
            require_close(checks["genetic_risk_revenue"], 21.653, 0.02),
            require_close(checks["genetic_risk_cvar"], -1.390, 0.03),
            checks["genetic_risk_loss_days"] == 111,
            checks["genetic_vs_df_revenue_wins"] == 237,
            checks["genetic_vs_df_revenue_losses"] == 240,
            require_close(checks["genetic_multiseed_revenue_mean"], 21.543, 0.02),
            require_close(checks["genetic_multiseed_revenue_std"], 0.078, 0.02),
            require_close(checks["genetic_multiseed_cvar_mean"], -1.530, 0.03),
            require_close(checks["genetic_multiseed_cvar_std"], 0.163, 0.03),
            checks["genetic_multiseed_seeds_above_df"] == 1,
            require_close(checks["pso_risk_revenue"], 21.596, 0.02),
            require_close(checks["pso_risk_cvar"], -1.741, 0.03),
            checks["pso_risk_loss_days"] == 113,
            checks["pso_vs_genetic_risk_wins"] == 173,
            checks["pso_vs_genetic_risk_losses"] == 194,
            require_close(checks["surrogate_risk_revenue"], 21.265, 0.03),
            require_close(checks["surrogate_risk_cvar"], -2.555, 0.04),
            checks["surrogate_risk_loss_days"] == 123,
            checks["surrogate_vs_genetic_risk_wins"] == 371,
            checks["surrogate_vs_genetic_risk_losses"] == 426,
            checks["surrogate_vs_q_learning_wins"] == 1272,
            checks["surrogate_vs_q_learning_losses"] == 106,
        ]
    )
    return checks


def verify_privacy_boundary() -> tuple[list[str], list[str]]:
    forbidden_path_terms = ["hunan", "shandong", "daily_disclosure", "user_load", "node_price"]
    allowed_content_terms = {"hunan", "shandong", "daily_disclosure", "user_load", "node_price"}
    flagged_paths = []
    flagged_content = []
    for path in SUPP.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(SUPP))
        lower = rel.lower()
        if any(term in lower for term in forbidden_path_terms):
            flagged_paths.append(rel)
        if path.suffix.lower() in {".md", ".txt", ".csv", ".py", ".sh"}:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if any(term in text for term in allowed_content_terms):
                # Manuscript discussion may mention local application boundaries; raw local data
                # paths and filenames are the critical packaging risk.
                flagged_content.append(rel)
    return flagged_paths, flagged_content


def verify_manuscript_claim_boundary() -> tuple[list[str], list[str]]:
    texts = []
    for path in (SUPP / "manuscript").glob("*.md"):
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    combined = "\n".join(texts)
    forbidden_found = [term for term in FORBIDDEN_MANUSCRIPT_CLAIMS if term in combined]
    required_missing = [term for term in REQUIRED_MANUSCRIPT_CLAIMS if term not in combined]
    return forbidden_found, required_missing


def main() -> None:
    rows, missing, mismatched = verify_manifest()
    zip_members, zip_uncompressed, zip_compressed, bad_zip_member = verify_zip()
    key = verify_key_results()
    privacy_path_flags, privacy_content_mentions = verify_privacy_boundary()
    forbidden_claims, missing_claims = verify_manuscript_claim_boundary()
    ok = (
        not missing
        and not mismatched
        and not bad_zip_member
        and not privacy_path_flags
        and not forbidden_claims
        and not missing_claims
        and key["review_required"] == 0
        and key["numeric_checks_pass"]
    )
    audit = f"""# Paper 3 Supplement Package Audit

Status: {'PASS' if ok else 'REVIEW REQUIRED'}

## Manifest

- Manifest rows: {len(rows)}
- Missing files: {len(missing)}
- Hash/size mismatches: {len(mismatched)}

## Zip Integrity

- Zip path: `{ZIP_PATH}`
- Zip members: {zip_members}
- Uncompressed bytes: {zip_uncompressed}
- Compressed bytes: {zip_compressed}
- First bad member from `ZipFile.testzip()`: `{bad_zip_member}`

## Key Result Checks

- Extended VPP risk simulator mean revenues: rolling-28d={key['risk_rolling_revenue']:.6f}, robust={key['risk_robust_revenue']:.6f}, prev-day={key['risk_prev_revenue']:.6f}.
- Held-out DF policy revenue: {key['policy_df_revenue']:.6f}.
- Forecast-coupled DF revenue/regret: {key['coupled_df_revenue']:.6f} / {key['coupled_df_regret']:.6f}.
- Risk sweep CVaR10 lambda=0.00 / 0.25: {key['risk_sweep_lambda0_cvar']:.6f} / {key['risk_sweep_lambda025_cvar']:.6f}.
- High-stress best revenue method: `{key['high_stress_best_revenue_method']}`.
- Fine-grid best revenue method: `{key['fine_best_revenue_method']}`.
- Fine-grid best CVaR method: `{key['fine_best_cvar_method']}`.
- Constrained Q-learning revenue/CVaR10: {key['q_learning_revenue']:.6f} / {key['q_learning_cvar']:.6f}.
- Constrained Q-learning losses versus DF revenue policy: {key['q_learning_vs_df_losses']} paired test days.
- Genetic risk-adjusted policy revenue/CVaR10/loss days: {key['genetic_risk_revenue']:.6f} / {key['genetic_risk_cvar']:.6f} / {key['genetic_risk_loss_days']}.
- Genetic risk-adjusted versus DF revenue wins/losses: {key['genetic_vs_df_revenue_wins']} / {key['genetic_vs_df_revenue_losses']} paired test days.
- Multi-seed genetic revenue mean/std: {key['genetic_multiseed_revenue_mean']:.6f} / {key['genetic_multiseed_revenue_std']:.6f}.
- Multi-seed genetic CVaR10 mean/std: {key['genetic_multiseed_cvar_mean']:.6f} / {key['genetic_multiseed_cvar_std']:.6f}.
- Multi-seed genetic seeds at or above DF revenue baseline: {key['genetic_multiseed_seeds_above_df']} of 8.
- PSO risk-adjusted policy revenue/CVaR10/loss days: {key['pso_risk_revenue']:.6f} / {key['pso_risk_cvar']:.6f} / {key['pso_risk_loss_days']}.
- PSO risk-adjusted versus genetic risk-adjusted wins/losses: {key['pso_vs_genetic_risk_wins']} / {key['pso_vs_genetic_risk_losses']} paired test days.
- Numeric/result checks pass: {key['numeric_checks_pass']}.
- Reference register: {key['doi_verified']} DOI-verified, {key['url_verified']} URL entries, {key['review_required']} review-required.

## Privacy Boundary

- Local/private-data filename flags: {len(privacy_path_flags)}
- Local/private-data content mentions in text-like files: {len(privacy_content_mentions)}

## Manuscript Claim Boundary

- Forbidden old implementation claims found: {len(forbidden_claims)}
- Required auditable-policy-search terms missing: {len(missing_claims)}

Flagged paths:

{chr(10).join('- `' + item + '`' for item in privacy_path_flags) if privacy_path_flags else '- None'}

Text files mentioning local-data terms:

{chr(10).join('- `' + item + '`' for item in privacy_content_mentions) if privacy_content_mentions else '- None'}

Forbidden claim findings:

{chr(10).join('- `' + item + '`' for item in forbidden_claims) if forbidden_claims else '- None'}

Missing required claim terms:

{chr(10).join('- `' + item + '`' for item in missing_claims) if missing_claims else '- None'}

## Interpretation

This audit verifies package structure, checksums, zip readability, key public result consistency, reference-register status, absence of local/private-data filenames in bundled paths, and consistency between the bundled manuscript copy and the current auditable policy-search claim boundary. Text mentions of local case-study boundaries are reported for human review but do not fail the package unless raw/private filenames are bundled as file paths.
"""
    AUDIT.write_text(audit, encoding="utf-8")
    print(AUDIT)
    print("PASS" if ok else "REVIEW REQUIRED")


if __name__ == "__main__":
    main()
