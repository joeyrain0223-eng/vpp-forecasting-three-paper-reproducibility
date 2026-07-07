from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

from create_submission_candidate_manuscripts import rebuild_docx


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASE = REPOSITORY_ROOT
ROOT = REPOSITORY_ROOT
PKG = REPOSITORY_ROOT / "paper2_ieee_access" / "paper2_transferable_load_reproducibility_package"
SOURCE = PKG / "manuscript" / "paper_2_transferable_load_forecasting.md"
OUT = PKG / "manuscript" / "ieee_access"
OUT.mkdir(parents=True, exist_ok=True)

AUTHOR = "zhijie REN"
AFFILIATION = "College of Computer Science, Hunan University, Lushan South Road, Yuelu District, Changsha, Hunan 410082, China"
EMAIL = "471062741@qq.com"
ORCID = "0009-0006-1048-6640"
TITLE = (
    "Transferable Short-Term Load Forecasting for Aggregated Virtual Power Plant "
    "Resources via Source-Pooled Time-Series Representation Learning"
)

IEEE_SUBMISSION_GUIDE = "https://ieeeaccess.ieee.org/authors/submission-guidelines/"
IEEE_TEMPLATE_URL = "https://ieeeaccess.ieee.org/wp-content/uploads/2025/08/Access-Template-2024.docx"
UCI_URL = "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014"
OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/"
UCI_APPLIANCES_URL = "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction"
SUPPLEMENT = ROOT / "submission_supplements" / "paper2_transferable_load_reproducibility_package.zip"
SUPPLEMENT_AUDIT = ROOT / "submission_supplements" / "paper2_transferable_load_reproducibility_package_audit.md"
SCHOOL_PACKET = OUT / "ieee_access_school_classification_confirmation_packet_2026-06-30.docx"
TEMPLATE_FORMATTED = OUT / "paper_2_ieee_access_template_formatted_manuscript.docx"
GITHUB_REPO = "https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility"
GITHUB_RELEASE = "https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/tag/v0.1.0-pre-doi"
GITHUB_ASSET = "https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/download/v0.1.0-pre-doi/three_paper_public_repository_staging_bundle.zip"
DOI_STATUS = "DOI pending: no Zenodo/Figshare/OSF DOI has been issued yet."

AI_DECLARATION = (
    "No conventional acknowledgements are made in this article; this section is "
    "included solely to satisfy IEEE AI-use disclosure requirements and contains "
    "no thanks to people, institutions, funders, or projects. During preparation "
    "of this manuscript, OpenAI ChatGPT/Codex was used to support manuscript "
    "organization, language editing, code/package documentation, and submission "
    "checklist drafting. The AI-assisted content was limited to drafting and "
    "editing support for manuscript text, documentation, and submission materials. "
    "The tool was not used as an author, did not determine the scientific "
    "conclusions, and did not replace verification of data, code, references, "
    "results, or claims. After using these tools, the author reviewed and edited "
    "all AI-assisted text, kept the disclosure consistent with IEEE policy, and "
    "takes full responsibility for the final content."
)

DATA_AVAILABILITY = (
    "The reproducible public-data layer uses the UCI Electricity Load Diagrams "
    "2011-2014 dataset, the Open Power System Data time-series package, and the "
    "UCI Appliances Energy Prediction dataset. The "
    "public reproducibility supplement contains processed public data, preprocessing "
    "scripts, result tables, generated figures, verified references, manuscript "
    "files, manifest hashes, and an audit report for the UCI multi-client transfer, "
    "source-pooled representation, masked-reconstruction representation, cold-start, client-level statistical-test, "
    "random-convolution representation checks, trainable dilated-convolution ridge "
    "checks, multi-seed TDConv stability checks, CPU-only patch-attention transfer checks, "
    "source-trained MLP transfer checks, neural TDConv residual-head checks, OPSD public load baseline, and "
    "external UCI Appliances one-hour-ahead and multi-horizon load sanity checks. "
    "Local Hunan and Shandong operational records are used only as non-public "
    "application-context evidence and are not redistributed. A public GitHub "
    f"pre-DOI release is available at {GITHUB_RELEASE}; a persistent DOI has not "
    "yet been issued and should be inserted only after a DOI-backed archive page exists."
)

DATA_AVAILABILITY_STATEMENT = (
    "The OPSD public benchmark [18] is reproducible from "
    "`public_data_download_templates.py` and `run_public_opsd_baselines.py`. "
    "The UCI multi-client transfer benchmark [17] is reproducible from "
    "`run_uci_load_transfer_baselines.py`; the masked-reconstruction "
    "representation prototype is reproducible from "
    "`run_uci_ssl_representation_prototype.py`; the cold-start/domain-shift "
    "diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; "
    "the client-level paired tests are reproducible from "
    "`run_uci_client_statistical_tests.py`; the random-convolution "
    "representation check is reproducible from "
    "`run_uci_random_conv_representation.py`; the trainable "
    "dilated-convolution ridge check is reproducible from "
    "`run_uci_trainable_tdconv_baseline.py`; the multi-seed TDConv stability check is reproducible from "
    "`run_uci_tdconv_multiseed_stability.py`; the CPU-only patch-attention transfer "
    "check is reproducible from `run_uci_patch_attention_transfer_baseline.py`; the source-trained MLP transfer "
    "check is reproducible from `run_uci_source_mlp_transfer_baseline.py`; and the neural TDConv residual-head "
    "check is reproducible from `run_uci_neural_tdconv_residual_check.py`. The second public load-dataset "
    "check on UCI Appliances Energy Prediction [21] is reproducible from "
    "`run_uci_appliances_energy_baselines.py`, and its 1-hour, 3-hour, 6-hour, "
    "and 12-hour robustness extension is reproducible from "
    "`run_uci_appliances_multihorizon_robustness.py`. Main claims are reproducible "
    "without relying solely on private data."
)


def root_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sanitize_upload_visible_audit_text(text: str) -> str:
    return (
        text.replace(str(ROOT), ".")
        .replace(str(BASE), "[local-workspace]")
        .replace(str(Path.home()), "[local-user]")
    )


EVIDENCE_MATCHED_ABSTRACT = """## Abstract

Accurate load forecasting is fundamental for virtual power plants that coordinate distributed resources, flexible demand, storage, and market transactions. However, aggregated virtual power plant resources often face cross-domain heterogeneity, limited historical observations, and cold-start deployment conditions. Conventional supervised load forecasting models can degrade when applied to new buildings, industrial parks, charging clusters, or regional aggregations. This paper studies transferable short-term load forecasting as a time-series representation-learning problem. Instead of presenting a single universal deep model, the reproducible framework evaluates source-pooled temporal representations under transparent transfer protocols. The checks include masked-reconstruction features, random-convolution features, trainable dilated-convolution ridge features, multi-seed TDConv stability, CPU-only patch-attention transfer, source-trained MLP transfer, and a NumPy neural residual head. Each representation is paired with source heads or lightweight target adapters. Public experiments use OPSD system-load zones, UCI Electricity Load Diagrams multi-client transfer, and UCI Appliances Energy Prediction external validation. The strongest UCI evidence comes from source-pooled temporal representations that improve held-out client RMSE against target-only ridge baselines; the multi-seed TDConv check confirms that this gain is stable to source-window subsampling, and the patch-attention check improves slightly over the TDConv adapter while target-only patch fitting and source-trained MLP transfer expose useful negative controls. The external appliance dataset shows that transparent lag-weather features can remain competitive. The study frames virtual power plant load forecasting as a computer-science problem of representation reuse, label efficiency, reproducibility, and failure-aware adaptation under domain shift.

Index Terms:"""


def align_claims_to_evidence(text: str) -> str:
    """Narrow Paper 2 claims to the methods actually supported by public experiments."""
    text = re.sub(
        r"## Abstract\s+.*?\n\s*(?:Index Terms:|Keywords:)",
        EVIDENCE_MATCHED_ABSTRACT,
        text,
        flags=re.S,
    )
    replacements = {
        "Self-supervised learning offers a promising route, consistent with recent time-series representation learning work [11]. Instead of requiring labels for every target resource, a model can learn temporal semantics from unlabeled load sequences through masked reconstruction, contrastive learning, temporal order prediction, or multi-resolution consistency. The learned encoder can then be adapted to downstream forecasting tasks with fewer labels. This paradigm has been successful in language and vision and is increasingly important for time-series learning.": (
            "Self-supervised and source-pooled representation learning offer practical routes for this setting, consistent with recent time-series representation learning work [11]. Instead of requiring labels for every target resource, a model can learn temporal features from source load archives through masked reconstruction, temporal-filter representations, or other pretext and representation objectives. The learned representation can then be adapted to downstream forecasting tasks with fewer labels. This paradigm is increasingly important for time-series learning, but it must be tested against strong target-only and transparent baselines rather than only against weak seasonal rules."
        ),
        "This paper proposes a self-supervised transferable load forecasting framework for aggregated virtual power plant resources. The method pretrains a time-series encoder on heterogeneous load data, adapts it using domain-specific normalization and lightweight adapters, and evaluates it under practical deployment scenarios: cross-building transfer, cross-region transfer, few-shot adaptation, and cold-start forecasting. The goal is not merely to reduce RMSE in a single dataset, but to improve generalization under distribution shift.": (
            "This paper studies a transferable load forecasting framework for aggregated virtual power plant resources. The method learns source-domain temporal representations, adapts them through lightweight target heads or adapters, and evaluates them under practical deployment scenarios: cross-client transfer, few-shot adaptation, and cold-start forecasting. The goal is not merely to reduce RMSE in a single dataset, but to understand when reusable representations improve generalization under distribution shift and when simple target baselines remain competitive."
        ),
        "### 4.3 Contrastive domain-aware objective\n\nThe model also uses a contrastive objective. Two augmented views of the same load window are treated as a positive pair, while windows from different times or domains serve as negatives. To avoid destroying domain-specific information, augmentations preserve daily shape and weather-load relationships. The contrastive loss encourages robust temporal representations while allowing downstream domain adaptation.": (
            "### 4.3 Domain-shift-aware representation checks\n\nThe reproducible implementation treats domain shift as an empirical object rather than as a solved property of the encoder. It compares masked-reconstruction features, random-convolution temporal features, and trainable dilated-convolution ridge features under the same source-head and target-adapter protocols. This design tests whether representation strength, target-label budget, and client mismatch explain transfer gains. Contrastive or adversarial extensions are therefore reserved as future reviewer-response options, not as claims required by the current public evidence."
        ),
        "Load magnitude and variability differ significantly across resources. A domain-adaptive normalization layer estimates scale and shift parameters for each resource or resource type. Lightweight adapters are inserted after encoder blocks and fine-tuned on target-domain data. This reduces the number of parameters that must be updated in few-shot settings and lowers overfitting risk.": (
            "Load magnitude and variability differ significantly across resources. The current protocol standardizes source and target windows, fits source-domain representation heads, and then uses lightweight target adapters with 1, 3, 7, or 28 days of target data. This reduces the number of target-fitted parameters and exposes low-label overfitting when the 1-day and 3-day adapters become unstable. The paper therefore claims regularized adaptation value under matched protocols rather than broad target-only refitting."
        ),
        "The model is trained in three stages:\n\n1. Self-supervised pretraining on unlabeled multi-domain load sequences.\n2. Supervised fine-tuning on source-domain forecasting labels.\n3. Target-domain adaptation with limited labeled data or metadata-only cold-start features.": (
            "The model family is evaluated in three stages:\n\n1. Source-domain representation fitting on multi-client load sequences.\n2. Source-head forecasting on held-out target clients with no target labels.\n3. Target-domain adaptation with limited labeled data and matched target-only baselines."
        ),
        "Ablations remove masked reconstruction, contrastive learning, domain-adaptive normalization, adapters, metadata embeddings, and pretraining. The analysis reports whether each component contributes more to within-domain accuracy or cross-domain generalization.": (
            "Ablations compare masked-reconstruction features, random-convolution features, trainable dilated-convolution ridge features, CPU-only patch-attention features, a source-trained MLP encoder, a neural residual-head extension, source heads, target adapters, and target-only negative controls. The analysis reports whether each component contributes more to cross-domain generalization, cold-start transfer, or ordinary target-domain accuracy."
        ),
        "This paper proposes a transferable short-term load forecasting framework for aggregated virtual power plant resources. By combining self-supervised temporal representation learning, domain-adaptive normalization, and lightweight adapters, the method is designed for cross-domain, few-shot, and cold-start settings. The paper contributes a computer-science framing of load forecasting as transferable time-series representation learning and provides a practical forecasting component for virtual power plant operations.": (
            "This paper presents a transferable short-term load forecasting study for aggregated virtual power plant resources. By comparing masked-reconstruction, random-convolution, and trainable dilated-convolution representations under matched source-head and target-adapter protocols, the work shows when source-pooled temporal features help cross-domain, few-shot, and cold-start load forecasting. The conclusion is deliberately bounded: representation reuse improves several held-out UCI client protocols and supports label-scarce adaptation, but representation choice, target-label budget, and client mismatch still require validation. The external UCI Appliances check further motivates a failure-aware interpretation, because transparent lag-weather models remain competitive when the public task is dominated by ordinary lag, calendar, and environmental structure."
        ),
        "This paper presents a transferable short-term load forecasting study for aggregated virtual power plant resources. By comparing masked-reconstruction, random-convolution, and trainable dilated-convolution representations under matched source-head and target-adapter protocols, the work shows when source-pooled temporal features help cross-domain, few-shot, and cold-start load forecasting. The conclusion is deliberately bounded: reusable representations improve several held-out UCI client protocols, but representation choice and target-label budget still require validation, and transparent lag-weather models remain competitive on the external UCI Appliances check.": (
            "This paper presents a transferable short-term load forecasting study for aggregated virtual power plant resources. By comparing masked-reconstruction, random-convolution, and trainable dilated-convolution representations under matched source-head and target-adapter protocols, the work shows when source-pooled temporal features help cross-domain, few-shot, and cold-start load forecasting. The conclusion is deliberately bounded: representation reuse improves several held-out UCI client protocols and supports label-scarce adaptation, but representation choice, target-label budget, and client mismatch still require validation. The external UCI Appliances check further motivates a failure-aware interpretation, because transparent lag-weather models remain competitive when the public task is dominated by ordinary lag, calendar, and environmental structure."
        ),
        "This paper presents a transferable short-term load forecasting study for aggregated virtual power plant resources. By comparing masked-reconstruction, random-convolution, and trainable dilated-convolution representations under matched source-head and target-adapter protocols, the work shows when source-pooled temporal features help cross-domain, few-shot, and cold-start load forecasting. The conclusion is deliberately bounded: reusable representations improve several held-out UCI client protocols, but representation choice and target-label budget still require validation, and transparent lag-weather models remain competitive on the external UCI Appliances one-hour and multi-horizon checks.": (
            "This paper presents a transferable short-term load forecasting study for aggregated virtual power plant resources. By comparing masked-reconstruction, random-convolution, and trainable dilated-convolution representations under matched source-head and target-adapter protocols, the work shows when source-pooled temporal features help cross-domain, few-shot, and cold-start load forecasting. The conclusion is deliberately bounded: representation reuse improves several held-out UCI client protocols and supports label-scarce adaptation, but representation choice, target-label budget, and client mismatch still require validation. The external UCI Appliances one-hour and multi-horizon checks further motivate a failure-aware interpretation, because transparent lag-weather and random-window models remain competitive when the public task is dominated by ordinary lag, calendar, and environmental structure."
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace(
        "low-label overfitting when very small adapters become unstable",
        "low-label overfitting when the 1-day and 3-day adapters become unstable",
    )
    text = text.replace(
        "but very small target-label adapters require stronger regularization or meta-validation",
        "but 1-day and 3-day target-label adapters require stronger regularization or meta-validation",
    )
    return text


def abstract_word_count(text: str) -> int:
    match = re.search(r"## Abstract\s+(.*?)\n\s*(?:Keywords:|Index Terms:)", text, re.S)
    if not match:
        return 0
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", match.group(1)))


def make_patch_attention_section() -> str:
    summary_path = ROOT / "public_experiment_results" / "uci_patch_attention_transfer_summary.csv"
    tests_path = ROOT / "public_experiment_results" / "uci_patch_attention_transfer_client_level_tests.csv"
    if not summary_path.exists() or not tests_path.exists():
        return ""
    with tests_path.open("r", encoding="utf-8", newline="") as f:
        tests = list(csv.DictReader(f))

    def comparison_row(name: str) -> dict:
        for row in tests:
            if row.get("comparison") == name:
                return row
        raise ValueError(f"missing comparison {name}")

    comparisons = [
        "PatchAttn 28d adapter vs TDConv 28d adapter",
        "PatchAttn source vs TDConv source",
        "PatchAttn 28d adapter vs RC 28d adapter",
        "PatchAttn 28d adapter vs target ridge 28d",
        "PatchAttn target head 28d vs target ridge 28d",
    ]
    labels = {
        "PatchAttn 28d adapter vs TDConv 28d adapter": "PatchAttn 28d vs TDConv 28d",
        "PatchAttn source vs TDConv source": "PatchAttn source vs TDConv source",
        "PatchAttn 28d adapter vs RC 28d adapter": "PatchAttn 28d vs RC 28d",
        "PatchAttn 28d adapter vs target ridge 28d": "PatchAttn 28d vs target ridge",
        "PatchAttn target head 28d vs target ridge 28d": "PatchAttn target-head vs target ridge",
    }
    lines = [
        "Table 15 adds a CPU-only patch-attention transfer check. The representation divides each 168-hour history window into seven 24-hour patches, uses the most recent patch as a deterministic query, pools patch profiles by similarity weights, and fits the same source-head and target-adapter protocol used by the other UCI transfer baselines. It is reported as a lightweight reviewer-response baseline, not as full PatchTST or TFT training.",
        "",
        "|Comparison|Baseline RMSE|Patch-attention RMSE|Mean RMSE gain|Wins/losses|Sign-test p|Interpretation|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in comparisons:
        row = comparison_row(name)
        gain = float(row["mean_rmse_gain_pct"])
        wins = int(float(row["wins"]))
        losses = int(float(row["losses"]))
        p = float(row["sign_test_p_two_sided"])
        if "target head" in name:
            interp = "negative control for target-only high-dimensional patch fitting"
        elif "TDConv 28d" in name:
            interp = "slightly stronger source-pooled patch evidence"
        elif "TDConv source" in name:
            interp = "source-head patch evidence"
        else:
            interp = "positive transfer evidence"
        lines.append(
            f"|{labels[name]}|{float(row['mean_baseline_rmse']):.3f}|{float(row['mean_candidate_rmse']):.3f}|{gain:.3f}%|{wins}/{losses}|{p:.3f}|{interp}|"
        )
    lines.extend(
        [
            "",
            "The patch-attention check strengthens the representation-learning argument without overclaiming a Transformer result. The 28-day patch-attention adapter reaches mean RMSE 64.761, slightly better than the TDConv 28-day adapter at 65.291, with 9/10 client-level wins and p=0.021. The target-only patch-attention head is much worse than the target ridge baseline, which is an important negative control: patch features help when learned from source clients and regularized through the adapter protocol, but high-dimensional patch fitting on limited target data is not safe by itself.",
            "",
            "![FIGURE 15. CPU-only patch-attention transfer check on UCI load forecasting.](figures/paper2_fig16_uci_patch_attention_transfer_baseline.png)",
            "",
        ]
    )
    return "\n".join(lines)


def make_source_mlp_section() -> str:
    summary_path = ROOT / "public_experiment_results" / "uci_source_mlp_transfer_summary.csv"
    tests_path = ROOT / "public_experiment_results" / "uci_source_mlp_transfer_client_level_tests.csv"
    diagnostics_path = ROOT / "public_experiment_results" / "uci_source_mlp_transfer_training_diagnostics.csv"
    if not summary_path.exists() or not tests_path.exists() or not diagnostics_path.exists():
        return ""
    with tests_path.open("r", encoding="utf-8", newline="") as f:
        tests = list(csv.DictReader(f))
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        summary = {row["model"]: row for row in csv.DictReader(f)}
    diagnostics = list(csv.DictReader(diagnostics_path.open("r", encoding="utf-8", newline="")))

    def comparison_row(name: str) -> dict:
        for row in tests:
            if row.get("comparison") == name:
                return row
        raise ValueError(f"missing comparison {name}")

    comparisons = [
        "SourceMLP 28d adapter vs patch-attention 28d",
        "SourceMLP 28d adapter vs TDConv 28d",
        "SourceMLP source vs TDConv source",
        "SourceMLP 28d adapter vs target ridge 28d",
        "SourceMLP hidden target head 28d vs target ridge 28d",
    ]
    labels = {
        "SourceMLP 28d adapter vs patch-attention 28d": "SourceMLP 28d vs PatchAttn 28d",
        "SourceMLP 28d adapter vs TDConv 28d": "SourceMLP 28d vs TDConv 28d",
        "SourceMLP source vs TDConv source": "SourceMLP source vs TDConv source",
        "SourceMLP 28d adapter vs target ridge 28d": "SourceMLP 28d vs target ridge",
        "SourceMLP hidden target head 28d vs target ridge 28d": "SourceMLP hidden-head vs target ridge",
    }
    best_epoch = int(float(diagnostics[-1]["selected_epoch"])) if diagnostics else 0
    val_rmse = float(summary["SourceMLP+adapter-28d"]["source_validation_rmse_normalized"])
    adapter_rmse = float(summary["SourceMLP+adapter-28d"]["mean_rmse"])
    source_rmse = float(summary["SourceMLP-source-head"]["mean_rmse"])
    lines = [
        "Table 16 reports a source-trained MLP transfer check. The model trains a one-hidden-layer neural encoder on pooled source-client windows using only NumPy, then evaluates the same source-head and target-adapter protocol. This is a deliberately stronger neural-capacity boundary than the ridge heads, but it remains CPU-reproducible and avoids claiming a GPU-trained Transformer.",
        "",
        "|Comparison|Baseline RMSE|SourceMLP RMSE|Mean RMSE gain|Wins/losses|Sign-test p|Interpretation|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in comparisons:
        row = comparison_row(name)
        gain = float(row["mean_rmse_gain_pct"])
        wins = int(float(row["wins"]))
        losses = int(float(row["losses"]))
        p = float(row["sign_test_p_two_sided"])
        if "patch-attention" in name or "TDConv" in name:
            interp = "negative neural-capacity boundary"
        elif "hidden target head" in name:
            interp = "frozen-hidden target fitting remains unstable"
        else:
            interp = "no reliable target-ridge advantage"
        lines.append(
            f"|{labels[name]}|{float(row['mean_baseline_rmse']):.3f}|{float(row['mean_candidate_rmse']):.3f}|{gain:.3f}%|{wins}/{losses}|{p:.3f}|{interp}|"
        )
    lines.extend(
        [
            "",
            f"The source-trained MLP is not a new best model. Its 28-day adapter reaches mean RMSE {adapter_rmse:.3f}, and the source-only head reaches {source_rmse:.3f}; both are worse than the patch-attention and TDConv source-pooled baselines. The result is nevertheless useful: a source-trained nonlinear encoder with validation RMSE {val_rmse:.3f} on normalized source windows and selected epoch {best_epoch} does not transfer safely by itself. This supports the paper's bounded claim that cross-client load transfer depends on representation regularization and adapter design, not simply on increasing neural capacity.",
            "",
            "![FIGURE 16. Source-trained MLP transfer boundary check on UCI load forecasting.](figures/paper2_fig17_uci_source_mlp_transfer_baseline.png)",
            "",
        ]
    )
    return "\n".join(lines)


def make_ieee_manuscript() -> str:
    text = SOURCE.read_text(encoding="utf-8").strip()
    text = re.sub(r"^# .+?\n+", "", text)
    text = re.sub(r"\n## 6\.2 Pilot Results on Local User Load Data\n.*?(?=\n## 7\. Conclusion)", "\n", text, flags=re.S)
    text = re.sub(r"\n## Local Case-Study Data Assets\n.*?(?=\n## |\Z)", "\n", text, flags=re.S)
    text = re.sub(r"\n## Empirical Plan Updated with Local Data\n.*?(?=\n## |\Z)", "\n", text, flags=re.S)
    text = text.replace("Keywords:", "Index Terms:")
    text = text.replace("![Figure", "![FIGURE")
    text = text.replace("Figure 1.", "FIGURE 1.")
    text = text.replace("Figure 2.", "FIGURE 2.")
    text = text.replace("Figure 3.", "FIGURE 3.")
    text = text.replace("Figure 4.", "FIGURE 4.")
    text = text.replace("Figure 5.", "FIGURE 5.")
    text = text.replace("Figure 6.", "FIGURE 6.")
    text = text.replace("Figure 7.", "FIGURE 7.")
    text = text.replace("Figure 8.", "FIGURE 8.")
    text = text.replace("Figure 9.", "FIGURE 9.")
    text = align_claims_to_evidence(text)
    text = text.replace(
        "The next neural residual check tests whether adding nonlinearity changes that conclusion. The next neural residual check tests whether adding nonlinearity changes that conclusion. The next neural residual check tests whether adding nonlinearity changes that conclusion.",
        "The next neural residual check tests whether adding nonlinearity changes that conclusion.",
    )
    text = text.replace(
        "This paper presents a transferable short-term load forecasting study for aggregated virtual power plant resources. By comparing masked-reconstruction, random-convolution, trainable dilated-convolution, and neural residual-head checks under matched source-head and target-adapter protocols, the work shows when source-pooled temporal features help cross-domain, few-shot, and cold-start load forecasting. The conclusion is deliberately bounded: representation reuse improves several held-out UCI client protocols and supports label-scarce adaptation, while the neural residual check shows that extra nonlinear capacity does not automatically improve over the regularized TDConv head; representation choice, target-label budget, and failure-aware interpretation still require validation, and transparent lag-weather models remain competitive on the external UCI Appliances one-hour and multi-horizon checks.",
        "This paper presents a transferable short-term load forecasting study for aggregated virtual power plant resources. By comparing masked-reconstruction, random-convolution, trainable dilated-convolution, multi-seed TDConv stability, CPU-only patch-attention transfer, source-trained MLP transfer, and neural residual-head checks under matched source-head and target-adapter protocols, the work shows when source-pooled temporal features help cross-domain, few-shot, and cold-start load forecasting. The conclusion is deliberately bounded: representation reuse improves several held-out UCI client protocols and supports label-scarce adaptation; the multi-seed TDConv check shows that the strongest public representation result is stable to source-window subsampling; the patch-attention check provides a stronger source-pooled reviewer baseline while target-only patch fitting and source-trained MLP transfer expose unsafe capacity expansion; the neural residual check shows that extra nonlinear capacity does not automatically improve over the regularized TDConv head; representation choice, target-label budget, and failure-aware interpretation still require validation, and transparent lag-weather models remain competitive on the external UCI Appliances one-hour and multi-horizon checks.",
    )
    text = text.replace(
        "The OPSD public benchmark [18] is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark [17] is reproducible from `run_uci_load_transfer_baselines.py`; the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`; the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`; the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`; and the neural TDConv residual-head check is reproducible from `run_uci_neural_tdconv_residual_check.py`. The second public load-dataset and multi-horizon checks on UCI Appliances Energy Prediction [21] are reproducible from `run_uci_appliances_energy_baselines.py` and `run_uci_appliances_multihorizon_robustness.py`. Main claims are reproducible without relying solely on private data.",
        "The OPSD public benchmark [18] is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark [17] is reproducible from `run_uci_load_transfer_baselines.py`; the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`; the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`; the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`; the multi-seed TDConv stability check is reproducible from `run_uci_tdconv_multiseed_stability.py`; the CPU-only patch-attention transfer check is reproducible from `run_uci_patch_attention_transfer_baseline.py`; the source-trained MLP transfer check is reproducible from `run_uci_source_mlp_transfer_baseline.py`; and the neural TDConv residual-head check is reproducible from `run_uci_neural_tdconv_residual_check.py`. The second public load-dataset and multi-horizon checks on UCI Appliances Energy Prediction [21] are reproducible from `run_uci_appliances_energy_baselines.py` and `run_uci_appliances_multihorizon_robustness.py`. Main claims are reproducible without relying solely on private data.",
    )
    tdconv_stability_section = """Table 14 reports a multi-seed stability check for the strongest trainable TDConv representation. The experiment repeats source-window subsampling with eight random seeds while preserving the same source-client set, target-client holdout, chronological split, TDConv feature construction, ridge penalty, and 28-day target-adapter protocol. This test asks whether the TDConv advantage is an artifact of one favorable source-window subsample or a stable representation-learning effect under repeated public-data reruns.

|Comparison|Baseline RMSE|TDConv RMSE mean +/- sd|TDConv RMSE range|Minimum wins|Maximum losses|p-value pattern|
|---|---:|---:|---:|---:|---:|---|
|TDConv 28d adapter vs RC 28d adapter|67.500|65.338 +/- 0.056|65.283-65.465|10/10|0/10|0.002 for each seed|
|TDConv 28d adapter vs target ridge 28d|82.782|65.338 +/- 0.056|65.283-65.465|9/10|1/10|0.021 for each seed|
|TDConv source head stability|-|65.665 +/- 0.112|65.478-65.827|-|-|descriptive stability check|

Across all eight seeds, the 28-day TDConv adapter remains tightly concentrated between 65.283 and 65.465 mean RMSE. Every seed beats the random-convolution 28-day adapter on 10/10 target clients, and every seed beats the 28-day target-only ridge on 9/10 target clients. This closes an important reproducibility gap in the Paper 2 evidence: the strongest representation result is not a single stochastic run. The result still should not be overread as a universal neural architecture claim, because the neural residual-head extension remains neutral against the parsimonious TDConv ridge head.

![FIGURE 14. Multi-seed stability of trainable TDConv source-window subsampling.](figures/paper2_fig15_uci_tdconv_multiseed_stability.png)

"""
    patch_attention_section = make_patch_attention_section()
    source_mlp_section = make_source_mlp_section()
    text = text.replace(
        "![FIGURE 13. Multi-horizon robustness check on UCI Appliances Energy Prediction.](figures/paper2_fig13_uci_appliances_multihorizon_robustness.png)\n",
        "![FIGURE 13. Multi-horizon robustness check on UCI Appliances Energy Prediction.](figures/paper2_fig13_uci_appliances_multihorizon_robustness.png)\n\n"
        + tdconv_stability_section
        + patch_attention_section
        + source_mlp_section,
    )
    text = text.replace(
        "\nTable 12 adds a second public load dataset",
        "\n<!-- pagebreak -->\n\nTable 12 adds a second public load dataset",
    )
    text = text.replace(
        "Short-term load forecasting has been studied with statistical models, machine-learning regressors, deep neural networks, and energy-forecasting competition protocols [14], [15], [16]. Classical methods are reliable for stable aggregate loads, while deep models can capture nonlinear relationships among weather, calendar, and behavior. However, most supervised approaches assume that training and deployment data share similar distributions. In virtual power plant portfolios, this assumption is often violated.",
        "Short-term load forecasting has been studied with statistical models, machine-learning regressors, deep neural networks, and energy-forecasting competition protocols [14], [15], [16]. Classical methods are reliable for stable aggregate loads, while deep models can capture nonlinear relationships among weather, calendar, and behavior. Because virtual power plant operation also couples load forecasts with electricity-price forecasts and uncertainty-aware downstream decisions, the public OPSD layer is kept compatible with electricity-price benchmark practice and conformal-calibration literature [19], [20]. However, most supervised approaches assume that training and deployment data share similar distributions. In virtual power plant portfolios, this assumption is often violated.",
    )

    front = f"""# {TITLE}

{AUTHOR}

{AFFILIATION}

Corresponding author: {AUTHOR} (e-mail: {EMAIL}).

Funding: This research received no external funding.

Article type for IEEE Access submission system: Research Article.

"""

    compliance = f"""## Acknowledgment

{AI_DECLARATION}

## Data and Code Availability

{DATA_AVAILABILITY}

Public source URLs:

- UCI Electricity Load Diagrams 2011-2014: {UCI_URL}
- Open Power System Data time series package: {OPSD_URL}
- UCI Appliances Energy Prediction: {UCI_APPLIANCES_URL}

"""
    text = text.replace("## References", compliance + "## References")
    author_bio_heading = "## " + "Author Biography"
    text += f"""

{author_bio_heading}

{AUTHOR} (ORCID: {ORCID}) received training in electronic information and is currently a doctoral student with the College of Computer Science, Hunan University, Changsha, Hunan, China. His research interests include machine learning for energy time series, electricity price forecasting, short-term load forecasting, time-series representation learning, and decision-focused virtual power plant operation.
"""
    return front + text + "\n"


def cover_letter() -> str:
    return f"""# IEEE Access Paper 2 Cover Letter

Dear Editor,

I submit the manuscript "{TITLE}" for consideration as a Research Article in IEEE Access.

The manuscript studies short-term load forecasting for aggregated virtual power plant resources as a transferable time-series representation-learning problem. The engineering setting is virtual power plant operation, but the central contribution is computer-science oriented: evidence-matched temporal representation reuse, cross-client transfer, label-scarce adaptation, cold-start deployment, neural-capacity negative controls, and client-level statistical testing under domain shift. The manuscript deliberately avoids claiming a universal deep model; the public evidence compares masked-reconstruction, random-convolution, trainable dilated-convolution ridge, patch-attention, source-trained MLP, and neural TDConv residual-head checks under matched adaptation protocols.

The current evidence package uses public, reproducible data. It combines the UCI Electricity Load Diagrams 2011-2014 multi-client benchmark with an OPSD public load benchmark and an external UCI Appliances Energy Prediction load sanity check, while keeping local Hunan/Shandong operational records outside the public reproducibility layer. On the UCI target-client split, the zero-label source representation and 28-day adapter each beat the 28-day target-only ridge baseline on 9/10 held-out clients, the random-convolution representation check improves the 28-day adapter mean RMSE to 67.50 with 10/10 wins over the masked-reconstruction adapter, and the trainable dilated-convolution ridge check further improves mean RMSE to 65.29 with 10/10 wins over the random-convolution adapter. A multi-seed TDConv stability check reaches 65.338 +/- 0.056 mean RMSE across eight source-window subsampling seeds, with every seed beating the random-convolution adapter on 10/10 clients and the 28-day target ridge on 9/10 clients. A CPU-only patch-attention transfer check reaches mean RMSE 64.76, wins against the TDConv 28-day adapter on 9/10 clients, and exposes target-only patch fitting as a negative control. A source-trained MLP transfer check reaches mean RMSE 86.06 for the 28-day adapter and loses against patch-attention and TDConv on 10/10 clients, showing that extra neural capacity alone does not solve cross-client transfer. A NumPy neural TDConv residual head reaches mean RMSE 65.58, is neutral against the regularized TDConv adapter at 5/10 wins, but still beats the random-convolution adapter on 10/10 clients and the 28-day target ridge on 9/10 clients. On the UCI Appliances one-hour-ahead task, a transparent lag-weather ridge baseline reduces RMSE to 78.59 versus 98.98 for current-load persistence and 107.04 for a 24-hour seasonal baseline.

The manuscript is original, is not under consideration elsewhere, and has no external funding. The author has reviewed the IEEE Access AI-use disclosure wording in this draft package and will verify that the submitted Word/LaTeX source and portal-generated PDF match exactly before confirming real upload.

Sincerely,

{AUTHOR}

{AFFILIATION}

{EMAIL}

ORCID: {ORCID}
"""


def declarations() -> str:
    return f"""# IEEE Access Paper 2 Author Declarations

## Manuscript

{TITLE}

## Author and Contact

- Author: {AUTHOR}
- Affiliation: {AFFILIATION}
- Corresponding email: {EMAIL}
- ORCID: {ORCID}
- Funding: This research received no external funding.
- Conventional acknowledgements: None.

## Competing Interests

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgment / AI-Use Statement for IEEE Access Review

{AI_DECLARATION}

IEEE publication guidance requires AI-generated or AI-assisted text to be disclosed in the acknowledgments section when required by the workflow, and the IEEE Editorial Style Manual uses the singular heading "Acknowledgment." Because the author requested no conventional acknowledgements, the current manuscript uses the Acknowledgment section only for the mandatory AI-use disclosure and includes no thanks to people, institutions, funders, or projects. Before formal upload, confirm the final placement against the live IEEE Author Portal.

## Data and Code Availability

{DATA_AVAILABILITY}

## Human Approval Gate

Before real upload, confirm:

1. The author has reviewed all AI-assisted text.
2. The paper is not submitted or under review elsewhere.
3. All references are relevant, accurate, and not known to be retracted.
4. The public-data scripts and result tables reproduce the numerical claims.
5. The IEEE Access template, PDF, source file, biography, and ORCID requirements are satisfied.
"""


def data_code_pack() -> str:
    audit_text = SUPPLEMENT_AUDIT.read_text(encoding="utf-8") if SUPPLEMENT_AUDIT.exists() else "Audit file not generated yet."
    audit_text = sanitize_upload_visible_audit_text(audit_text)
    supplement_status = "Present" if SUPPLEMENT.exists() else "Missing"
    supplement_zip = root_relative(SUPPLEMENT)
    supplement_audit = root_relative(SUPPLEMENT_AUDIT)
    supplement_dir = supplement_zip.removesuffix(".zip")
    return f"""# IEEE Access Paper 2 Data and Code Availability Pack

## Public Reproducibility Layer

Primary public datasets:

- UCI Electricity Load Diagrams 2011-2014: {UCI_URL}
- Open Power System Data time-series package: {OPSD_URL}
- UCI Appliances Energy Prediction: {UCI_APPLIANCES_URL}

Main reproducibility scripts:

- `work/run_uci_load_transfer_baselines.py`
- `work/run_uci_ssl_representation_prototype.py`
- `work/run_uci_ssl_cold_start_diagnostics.py`
- `work/run_uci_client_statistical_tests.py`
- `work/run_uci_random_conv_representation.py`
- `work/run_uci_trainable_tdconv_baseline.py`
- `work/run_uci_tdconv_multiseed_stability.py`
- `work/run_uci_patch_attention_transfer_baseline.py`
- `work/run_uci_source_mlp_transfer_baseline.py`
- `work/run_uci_neural_tdconv_residual_check.py`
- `work/run_uci_appliances_energy_baselines.py`
- `work/run_public_opsd_baselines.py`

Primary result files:

- `public_experiment_results/uci_load_transfer_summary.csv`
- `public_experiment_results/uci_ssl_representation_summary.csv`
- `public_experiment_results/uci_ssl_cold_start_summary.csv`
- `public_experiment_results/uci_ssl_client_level_stat_tests.csv`
- `public_experiment_results/uci_random_conv_representation_summary.csv`
- `public_experiment_results/uci_random_conv_client_level_tests.csv`
- `public_experiment_results/uci_trainable_tdconv_baseline_summary.csv`
- `public_experiment_results/uci_trainable_tdconv_client_level_tests.csv`
- `public_experiment_results/uci_tdconv_multiseed_stability_results.csv`
- `public_experiment_results/uci_tdconv_multiseed_stability_by_seed.csv`
- `public_experiment_results/uci_tdconv_multiseed_stability_summary.csv`
- `public_experiment_results/uci_tdconv_multiseed_stability_tests.csv`
- `public_experiment_results/uci_patch_attention_transfer_summary.csv`
- `public_experiment_results/uci_patch_attention_transfer_client_level_tests.csv`
- `public_experiment_results/uci_source_mlp_transfer_summary.csv`
- `public_experiment_results/uci_source_mlp_transfer_client_level_tests.csv`
- `public_experiment_results/uci_source_mlp_transfer_training_diagnostics.csv`
- `public_experiment_results/uci_neural_tdconv_residual_summary.csv`
- `public_experiment_results/uci_neural_tdconv_residual_client_level_tests.csv`
- `public_experiment_results/uci_neural_tdconv_residual_training_diagnostics.csv`
- `public_experiment_results/uci_appliances_energy_dataset_stats.csv`
- `public_experiment_results/uci_appliances_energy_baselines.csv`

Generated supplement:

- Package directory: `{supplement_dir}/`
- Zip archive: `{supplement_zip}`
- Audit report: `{supplement_audit}`
- Current zip status: {supplement_status}

## Non-Public Boundary

Local Hunan and Shandong files are not included in a public release unless the author obtains explicit permission. They are treated only as application-context evidence and must not be uploaded as raw supplementary data.

## Recommended First-Submission Wording

The public-data layer of this study is reproducible from the UCI Electricity Load Diagrams 2011-2014 dataset, the Open Power System Data time-series package, and the UCI Appliances Energy Prediction dataset. The public supplement contains processed public data, preprocessing scripts, result tables, generated figures, verified references, manuscript files, manifest hashes, and an audit report for the main transfer-learning claims, including masked-reconstruction, random-convolution, trainable dilated-convolution ridge, multi-seed TDConv stability, CPU-only patch-attention transfer, source-trained MLP transfer, neural TDConv residual-head, and external UCI Appliances load checks. Local Hunan and Shandong operational records are used only as non-public application-context evidence and are not redistributed.

## Portal-Ready Data and Code Wording

### Data Availability Statement

{DATA_AVAILABILITY_STATEMENT}

### Data and Code Availability

{DATA_AVAILABILITY}

## Repository Route

For the first IEEE Access submission, upload `paper2_transferable_load_reproducibility_package.zip` as supplementary material if the portal allows. A public GitHub pre-DOI release also exists and can be named in the portal if the final author-approved route allows a public repository URL before DOI assignment.

- GitHub repository: {GITHUB_REPO}
- GitHub release: {GITHUB_RELEASE}
- Release asset: {GITHUB_ASSET}
- Commit traceability: see the GitHub release receipt and repository history for the exact commit state used by the current pre-DOI release asset.
- DOI status: {DOI_STATUS}

Use the GitHub release URL as the current repository URL. Do not write a DOI until a Zenodo, Figshare, OSF, or equivalent DOI page exists.

## Current Audit Summary

```text
{audit_text.strip()}
```
"""


def checklist() -> str:
    wc = abstract_word_count(make_ieee_manuscript())
    return f"""# IEEE Access Paper 2 Final Upload Checklist

Generated: 2026-06-30

Target journal: IEEE Access.

Official working sources:

- IEEE Access submission guidelines: {IEEE_SUBMISSION_GUIDE}
- IEEE Access Word template: {IEEE_TEMPLATE_URL}

## Files Built in This Pack

- paper_2_ieee_access_manuscript_candidate.docx
- paper_2_ieee_access_cover_letter.docx
- paper_2_ieee_access_author_declarations.docx
- paper_2_ieee_access_data_code_availability_pack.docx
- paper2_transferable_load_reproducibility_package.zip, if generated
- ieee_access_school_classification_confirmation_packet_2026-06-30.docx, if generated separately
- paper_2_ieee_access_template_formatted_manuscript.docx, if generated separately from the official Access template

## Current Metadata

- Author: {AUTHOR}
- Affiliation: {AFFILIATION}
- Corresponding email: {EMAIL}
- Funding: no external funding.
- Conventional acknowledgements: none.
- AI-use disclosure: placed under the IEEE-style Acknowledgment heading solely as a compliance disclosure; it is not a conventional thanks section and contains no thanks to people, institutions, funders, or projects.

## IEEE Access Gate Check

| Gate | Current status | Action before real upload |
|---|---|---|
| Article type | Research Article recommended | Confirm in IEEE Author Portal |
| Official template file | {'Downloaded to templates/Access-Template-2024.docx' if (ROOT / 'templates' / 'Access-Template-2024.docx').exists() else 'Official template URL recorded'} | Recheck template URL immediately before real upload |
| Template-formatted manuscript | {'Present and rendered' if TEMPLATE_FORMATTED.exists() else 'Generate with work/build_paper2_ieee_access_template_formatted.py'} | Author must final-check Word source, PDF export, and portal proof |
| Word/PDF match | Template-formatted candidate generated and rendered if present | Compare final portal PDF against source after all author metadata is finalized |
| Abstract length | {wc} words | Keep between 150 and 250 words |
| Author list | Single author filled | Confirm no omitted contributors |
| ORCID | {ORCID} supplied by author | Verify visible populated ORCID in portal |
| Biography | Minimal biography draft included | Author must expand/confirm biography and photo requirement |
| AI disclosure | Placed under the IEEE-style Acknowledgment heading as a compliance disclosure, not as conventional thanks | Author/institution must approve wording and IEEE portal placement |
| Funding | No external funding | Keep consistent in portal fields |
| Data/code | Public supplement {'present and audited' if SUPPLEMENT.exists() and SUPPLEMENT_AUDIT.exists() else 'not fully generated'} | Upload supplement zip or choose repository/DOI route |
| School classification packet | {'Present' if SCHOOL_PACKET.exists() else 'Generate with work/build_paper2_ieee_access_school_confirmation_packet.py'} | Obtain written/screenshot confirmation for IEEE Access before APC payment or formal reliance |
| References | Verified reference assets exist | Final IEEE style and retraction check still required |

## Do Not Upload Until

1. The template-formatted manuscript is final-checked after biography/photo, ORCID portal-display verification, and disclosure wording are finalized.
2. The Word/LaTeX source and portal-generated PDF match exactly after the final upload proof is generated.
3. The author ORCID is verified in the portal and the biography/photo requirements are satisfied.
4. The AI disclosure wording is approved by the author and institution.
5. The public-data reproducibility package is either uploaded as supplement or deposited in a stable repository.
6. The school confirms IEEE Access satisfies the needed C-class route for the current graduation rule year.
"""


def readme() -> str:
    return f"""# IEEE Access Paper 2 Submission Candidate Pack

This directory contains the target-journal working package for Paper 2, the stable C-class route manuscript on transferable load forecasting for virtual power plant resources.

## Built Artifacts

- `paper_2_ieee_access_manuscript_candidate.md` / `.docx`
- `paper_2_ieee_access_cover_letter.md` / `.docx`
- `paper_2_ieee_access_author_declarations.md` / `.docx`
- `paper_2_ieee_access_data_code_availability_pack.md` / `.docx`
- `paper_2_ieee_access_final_upload_checklist.md` / `.docx`
- `paper2_transferable_load_reproducibility_package.zip`, if generated
- `ieee_access_school_classification_confirmation_packet_2026-06-30.md` / `.docx`, if generated separately
- `paper_2_ieee_access_template_formatted_manuscript.md` / `.docx`, if generated separately from the official Access template

## Official Sources to Recheck Before Upload

- IEEE Access submission guidelines: {IEEE_SUBMISSION_GUIDE}
- IEEE Access Word template: {IEEE_TEMPLATE_URL}

## Current Status

The package is a target-journal working package, not yet a formal upload package. The Paper 2 public reproducibility supplement has been generated and can be uploaded as supplementary material if the portal allows. The Chinese IEEE Access school-classification confirmation packet has also been generated separately when present in this folder. A template-formatted manuscript file has been generated from the downloaded IEEE Access Word template when `paper_2_ieee_access_template_formatted_manuscript.docx` is present; it uses IEEE Access styles in a single-column source layout to keep tables and figures readable for pre-upload review. The author ORCID is supplied as 0009-0006-1048-6640. The remaining hard gates are biography/photo confirmation, final reference style/retraction check, AI disclosure approval, repository/DOI route execution, final portal source/PDF proof comparison, and written/screenshot school confirmation for IEEE Access. Because IEEE Access is not an IEEE/ACM Transactions-series journal, it should not be treated as a Transactions-route B-class paper unless the school explicitly confirms that interpretation.
"""


FILES = {
    "paper_2_ieee_access_manuscript_candidate.md": make_ieee_manuscript,
    "paper_2_ieee_access_cover_letter.md": cover_letter,
    "paper_2_ieee_access_author_declarations.md": declarations,
    "paper_2_ieee_access_data_code_availability_pack.md": data_code_pack,
    "paper_2_ieee_access_final_upload_checklist.md": checklist,
    "README.md": readme,
}


def write_docx(md_path: Path, text: str) -> None:
    # Keep Markdown source readable while preventing inline Markdown markers from
    # leaking into Word renderings produced by the lightweight local converter.
    docx_text = text.replace("`", "")
    rebuild_docx(md_path, docx_text)


def main() -> None:
    written = []
    for name, maker in FILES.items():
        path = OUT / name
        text = maker()
        path.write_text(text, encoding="utf-8")
        written.append(path)
        if name.endswith(".md") and name != "README.md":
            write_docx(path, text)
            written.append(path.with_suffix(".docx"))

    template_dest = OUT / "Access-Template-2024.docx"
    existing = ROOT / "templates" / "Access-Template-2024.docx"
    if existing.exists():
        shutil.copy2(existing, template_dest)
        written.append(template_dest)
    if SUPPLEMENT.exists():
        dest = OUT / SUPPLEMENT.name
        shutil.copy2(SUPPLEMENT, dest)
        written.append(dest)

    print("\n".join(str(p) for p in written))


if __name__ == "__main__":
    main()
