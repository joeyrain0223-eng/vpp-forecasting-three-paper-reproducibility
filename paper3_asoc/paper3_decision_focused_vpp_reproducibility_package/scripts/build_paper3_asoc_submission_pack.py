from __future__ import annotations

import json
import re
import shutil
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from create_submission_candidate_manuscripts import rebuild_docx


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASE = REPOSITORY_ROOT
ROOT = REPOSITORY_ROOT
PKG = REPOSITORY_ROOT / "paper3_asoc" / "paper3_decision_focused_vpp_reproducibility_package"
SOURCE = PKG / "manuscript" / "paper_3_decision_focused_vpp_bidding.md"
OUT = PKG / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

AUTHOR = "zhijie REN"
AFFILIATION = "College of Computer Science, Hunan University"
ADDRESS = "College of Computer Science, Lushan South Road, Yuelu District, Changsha, Hunan 410082, China"
COVER_ADDRESS = "Lushan South Road, Yuelu District, Changsha, Hunan 410082, China"
EMAIL = "471062741@qq.com"
ORCID = "0009-0006-1048-6640"
TITLE = "Decision-Focused Learning for Virtual Power Plant Bidding under Electricity Price and Load Uncertainty"
ASOC_GUIDE = "https://www.sciencedirect.com/journal/applied-soft-computing/publish/guide-for-authors"
ASOC_SCOPE = "https://www.sciencedirect.com/journal/applied-soft-computing/about/aims-and-scope"
SUPPLEMENT = ROOT / "submission_supplements" / "paper3_decision_focused_vpp_reproducibility_package.zip"
SUPPLEMENT_AUDIT = ROOT / "submission_supplements" / "paper3_decision_focused_vpp_reproducibility_package_audit.md"
GRAPHICAL_ABSTRACT = OUT / "paper_3_asoc_graphical_abstract.png"
SCHOOL_PACKET = OUT / "asoc_school_classification_confirmation_packet_2026-06-30.docx"
GITHUB_REPO = "https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility"
GITHUB_RELEASE = "https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/tag/v0.1.0-pre-doi"
GITHUB_ASSET = "https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/download/v0.1.0-pre-doi/three_paper_public_repository_staging_bundle.zip"
DOI_STATUS = "DOI pending: no Zenodo/Figshare/OSF DOI has been issued yet."

AI_DECLARATION = (
    "During the preparation of this work, the author used OpenAI ChatGPT/Codex "
    "to support manuscript organization, language editing, code documentation, "
    "and submission-checklist drafting. After using these tools, the author "
    "reviewed and edited the content as needed and takes full responsibility for "
    "the content of the submitted manuscript. The tools were not used as authors, "
    "did not determine the scientific conclusions, and were not used to create or "
    "alter figures, images, or graphical abstract artwork."
)

DATA_STATEMENT = (
    "The public reproducibility layer is based on the Open Power System Data "
    "time-series package and is bundled as a public-data-only supplementary "
    "package containing processed public data, scripts, result CSV files, figures, "
    "reference assets, requirements, manifest hashes, and an audit report. The "
    "first submission may upload this package as supplementary material. A public "
    f"GitHub pre-DOI release is available at {GITHUB_RELEASE}; the same release "
    "should be archived through Zenodo, Figshare, OSF, or an equivalent citable "
    "repository before a DOI is inserted in the final article. Local Hunan "
    "and Shandong operational records are not redistributed because they are "
    "non-public application-context data. The GIS infrastructure-context audit "
    "uses local copies of open or externally sourced grid/resource metadata only "
    "as background evidence; raw OSM/GEM/GIS files are not included in the journal "
    "supplement until source-page, attribution, and redistribution gates are "
    "confirmed."
)

HIGHLIGHTS = [
    "Decision-focused VPP bidding aligns forecasts with market value.",
    "OPSD simulator reports revenue, regret, CVaR, and loss-day risk.",
    "Multi-seed genetic search tests stochastic policy-search stability.",
    "Forecast-coupled policy reaches 31.43 EUR/day and 4.67 regret.",
]


def root_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sanitize_upload_visible_audit_text(text: str) -> str:
    return (
        text.replace(str(ROOT), ".")
        .replace(str(BASE), "[local-workspace]")
        .replace(str(Path.home()), "[local-user]")
    )


def abstract_word_count(text: str) -> int:
    match = re.search(r"## Abstract\s+(.*?)\n\s*Keywords:", text, re.S)
    if not match:
        return 0
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", match.group(1)))


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_abstract_and_keywords(text: str) -> tuple[str, list[str]]:
    match = re.search(r"^##\s+Abstract\s*\n(.*?)(?=^##\s+|\Z)", text, re.S | re.M | re.I)
    block = match.group(1).strip() if match else ""
    key_match = re.search(r"(?:\*\*)?\b(?:Keywords|Index Terms):(?:\*\*)?\s*(.+)$", block, re.I)
    if not key_match:
        return compact(block), []
    abstract = compact(block[: key_match.start()])
    raw = key_match.group(1).strip().rstrip(".").strip("* ")
    keywords = [compact(item).strip("* ") for item in re.split(r";|,", raw) if compact(item).strip("* ")]
    return abstract, keywords


def char_count(s: str) -> int:
    return len(s)


def strip_local_sections(text: str) -> str:
    for heading in ["## Local Case-Study Data Assets", "## Empirical Plan Updated with Local Data"]:
        pattern = rf"\n{re.escape(heading)}\n.*?(?=\n## |\Z)"
        text = re.sub(pattern, "\n", text, flags=re.S)
    return text


def normalize_for_asoc() -> str:
    text = SOURCE.read_text(encoding="utf-8").strip()
    text = re.sub(r"^# .+?\n+", "", text)
    text = strip_local_sections(text)
    text = re.sub(
        r"## Abstract\s+.*?\n\s*Keywords:",
        """## Abstract

Virtual power plant bidding decisions depend on forecasts of electricity price, load, renewable generation, and resource flexibility. Standard forecasting pipelines optimize prediction accuracy first and then pass forecasts into a separate optimization module. This forecast-then-optimize paradigm can be suboptimal because the forecasting loss may not align with downstream market profit, risk exposure, or operational penalties. This paper studies a decision-focused, public-data virtual power plant bidding protocol under electricity price and load uncertainty. Rather than claiming a full neural differentiable market engine, the implemented framework uses auditable forecast-to-decision simulators, trainable policy-search coefficients selected on historical decision value, forecast-coupled policies, fuzzy risk-aware scoring, genetic soft-computing policy search with multi-seed stability diagnostics, particle-swarm policy search as an independent swarm-intelligence check, a lightweight differentiable surrogate-policy baseline, constrained tabular Q-learning as a policy-learning stress test, and robustness/risk-aversion checks. Price and net-load uncertainty enter through public forecast signals, rolling robust baselines, fuzzy memberships, CVaR-oriented policy selection, and chronological learning baselines. The empirical evaluation compares previous-day forecast-then-optimize, rolling-mean forecast-then-optimize, robust quantile forecast-then-optimize, fuzzy risk-aware forecast-then-optimize, forecast-coupled policies, held-out decision-focused policy search, genetic policy search, particle-swarm policy search, surrogate policy-gradient search, constrained Q-learning, risk-aversion sweeps, and settlement-stress checks on public OPSD market scenarios. The study frames virtual power plant operation as a machine-learning problem of aligning prediction signals with downstream decision quality under transparent reproducibility constraints.

Keywords:""",
        text,
        flags=re.S,
    )
    text = text.replace(
        "The empirical evaluation compares previous-day forecast-then-optimize, rolling-mean forecast-then-optimize, robust quantile forecast-then-optimize, and held-out decision-focused policy search on public-market and local case-study scenarios.",
        "The empirical evaluation compares previous-day forecast-then-optimize, rolling-mean forecast-then-optimize, robust quantile forecast-then-optimize, forecast-coupled policies, and held-out decision-focused policy search on public OPSD market scenarios.",
    )
    text = text.replace(
        "The local disclosure-data pilot is retained as application evidence rather than the main reproducibility claim.",
        "The target-journal version keeps the empirical claims on the public reproducibility layer, while local application records are treated only as non-public deployment context.",
    )
    text = re.sub(r"\n!\[Figure \d+\..*?paper3_fig2_daily_regret\.png\)\n", "\n", text)
    text = re.sub(r"\n!\[Figure \d+\..*?paper3_fig3_revenue_regret\.png\)\n", "\n", text)
    text = re.sub(r"\n## 6\.2 Pilot Decision Simulation\n.*?(?=\n## Appendix|\n## Conclusion|\Z)", "\n", text, flags=re.S)
    text = text.replace("local fine-grid", "fine-grid")
    text = text.replace("Fine local revenue grid", "Fine revenue grid")
    text = text.replace("Fine local risk-adjusted grid", "Fine risk-adjusted grid")
    text = text.replace("local policy-grid refinement", "fine policy-grid refinement")
    text = text.replace("local refinement grid", "fine refinement grid")
    text = text.replace(
        "This paper proposes a decision-focused learning framework for virtual power plant bidding. The method combines probabilistic price and load forecasting with a bidding optimization layer that accounts for storage constraints, flexible-load limits, and risk preferences. The paper compares the proposed framework with standard forecast-then-optimize methods and evaluates not only prediction error but also revenue, regret, risk, and operational penalties.",
        "This paper studies decision-focused learning for virtual power plant bidding through a transparent public-data simulator and trainable policy-search layer. The implemented method combines probabilistic price and net-load forecast signals with a bidding simulator that accounts for storage constraints, flexible-load limits, imbalance penalties, and risk preferences. The paper compares this auditable decision-focused protocol with standard forecast-then-optimize methods and evaluates not only prediction error but also revenue, regret, risk, and operational penalties.",
    )
    text = text.replace(
        "The proposed method uses uncertainty not only for reporting prediction intervals but also as an input to risk-aware bidding decisions.",
        "The implemented protocol uses uncertainty not only for reporting prediction intervals but also as an input to risk-aware bidding decisions and policy selection.",
    )
    text = text.replace(
        "The model can therefore learn representations that prioritize errors with high operational consequences.",
        "Such methods can therefore prioritize prediction errors with high operational consequences when the implementation supports a decision-loss signal.",
    )
    text = text.replace(
        "The model does not optimize only expected revenue. A risk-aware decision layer uses CVaR or quantile-based downside constraints, following the standard CVaR optimization formulation [6]:",
        "The implemented protocol does not evaluate only expected revenue. A risk-aware decision layer uses CVaR or quantile-based downside constraints, following the standard CVaR optimization formulation [6]:",
    )
    text = re.sub(
        r"### 4\.3 Decision-focused training\n\n.*?(?=\n### 4\.4 Risk-aware scenario selection)",
        """### 4.3 Auditable decision-focused policy search

The reproducible implementation uses a transparent surrogate decision-focused layer rather than an end-to-end neural optimizer. Candidate policies score charge and discharge hours with historical price mean, price volatility, and net-load shape features. Policy coefficients are selected on chronological training and validation days by maximizing realized decision value or a revenue-plus-CVaR objective, and the selected coefficients are then frozen before held-out test evaluation.

This design tests the predict-and-optimize hypothesis without relying on an opaque proprietary market engine. The decision objective is:

L_decision = -J(u_theta; y) - lambda_r CVaR10(u_theta),

where J is realized daily decision value in the public simulator and lambda_r controls risk aversion. Differentiable optimization layers remain a natural extension, but the current manuscript claims only the public, auditable surrogate policy-search evidence reported in the experiments.

""",
        text,
        flags=re.S,
    )
    text = text.replace(
        "This paper proposes a decision-focused learning framework for virtual power plant bidding under electricity price and load uncertainty. By integrating probabilistic forecasting with an optimization-aware training objective, the method aligns predictive representations with market-operation outcomes. The empirical evaluation emphasizes revenue, regret, downside risk, and imbalance penalties, thereby moving beyond conventional forecasting accuracy. The work contributes to computer-science research on predict-and-optimize learning while addressing a practical virtual power plant decision problem.",
        "This paper studies decision-focused learning for virtual power plant bidding under electricity price and load uncertainty through a public, auditable forecast-to-decision simulator. By connecting probabilistic price and net-load forecast signals to policy-search objectives, the method evaluates predictive representations by market-operation outcomes rather than by forecasting error alone. The empirical evaluation emphasizes revenue, regret, downside risk, imbalance penalties, risk-aversion sensitivity, and settlement robustness. The work contributes bounded evidence for computer-science research on predict-and-optimize learning while addressing a practical virtual power plant decision problem.",
    )
    text = text.replace(
        "The contribution is not a new power-system dispatch formulation. Rather, the contribution is a learning framework that connects uncertain time-series prediction with downstream optimization objectives in a virtual power plant market setting.\n\n## 2. Related Work",
        "The contribution is not a new power-system dispatch formulation. Rather, the contribution is a learning framework that connects uncertain time-series prediction with downstream optimization objectives in a virtual power plant market setting.\n\n![Figure 1. Decision-focused VPP bidding framework connecting forecast signals, uncertainty, policy search, and settlement-risk evaluation.](figures/paper3_fig1_framework.png)\n\n## 2. Related Work",
    )
    figure_renumbering = {
        "Figure 4. OPSD VPP extended risk simulation: average daily revenue by implementable policy.": "Figure 2. OPSD VPP extended risk simulation: average daily revenue by implementable policy.",
        "Figure 5. OPSD VPP extended risk simulation: worst-decile revenue CVaR by implementable policy.": "Figure 3. OPSD VPP extended risk simulation: worst-decile revenue CVaR by implementable policy.",
        "Figure 6. OPSD decision-focused policy search on held-out public test days.": "Figure 4. OPSD decision-focused policy search on held-out public test days.",
        "Figure 7. Forecast-coupled OPSD VPP decision test linking Paper 1 and Paper 2 forecasts to Paper 3.": "Figure 5. Forecast-coupled OPSD VPP decision test linking Paper 1 and Paper 2 forecasts to Paper 3.",
        "Figure 8. OPSD risk-aversion sensitivity of decision-focused VPP policy search.": "Figure 6. OPSD risk-aversion sensitivity of decision-focused VPP policy search.",
        "Figure 9. OPSD reviewer robustness checks for settlement stress and fine policy-grid refinement.": "Figure 7. OPSD reviewer robustness checks for settlement stress and fine policy-grid refinement.",
    }
    for old_caption, new_caption in figure_renumbering.items():
        text = text.replace(old_caption, new_caption)
    text = text.replace(
        "Keywords: virtual power plant; electricity market bidding; decision-focused learning; predict-and-optimize; uncertainty; robust optimization; fuzzy risk-aware baseline; time-series forecasting.",
        "Keywords: virtual power plant; bidding; decision-focused learning; predict-and-optimize; uncertainty; fuzzy risk-aware baseline; CVaR.",
    )
    text = text.replace("## Data Availability Statement", "## Data Availability")
    genetic_section = """### 6.8 Constrained Q-learning policy-learning boundary

Table 10 adds a constrained tabular Q-learning baseline to address the natural reviewer question of whether a reinforcement-learning style policy learner should replace the auditable decision-focused policy search. The Q-learning baseline observes only rolling historical price, volatility, net-load-shape bins, hour block, and remaining charge/discharge counts. It is trained on chronological training and validation days, then frozen for the final 20 percent public test split. The result is deliberately reported even though it is negative.

Table 10 reports the constrained Q-learning boundary check on the final OPSD test split. The discretized policy improves over the no-action baseline but remains below the transparent forecast-then-optimize and continuous search policies, showing that policy-learning capacity alone is not sufficient under this information boundary.

|Method|Revenue|Regret|CVaR10|Loss days|Penalty|
|---|---:|---:|---:|---:|---:|
|DF policy search (revenue)|21.639|11.831|-1.853|115|3.392|
|Robust quantile FTO|21.128|12.342|-0.585|97|3.295|
|Fuzzy risk-aware FTO|11.342|22.128|-4.968|259|3.295|
|Constrained Q-learning policy|-5.655|39.125|-24.534|1034|3.352|

![Figure 8. Constrained tabular Q-learning exposes the policy-learning boundary on the OPSD VPP test split, where discretized value learning is benchmarked against transparent revenue and risk baselines without claiming neural-RL superiority.](figures/paper3_fig12_opsd_constrained_q_learning_vpp.png)

The constrained Q-learning policy loses to the revenue-selected decision-focused policy on 1280 of 1378 paired test days, with a mean revenue deficit of 27.379 EUR/day and an exact two-sided sign-test p-value below 1e-250. The result is not hidden because it clarifies the scope of the contribution: under the current public simulator and compact state representation, a transparent coefficient-search policy is more reliable than a tabular policy learner.

### 6.9 Genetic soft-computing policy search

Table 11 adds a genetic soft-computing policy-search baseline to test whether continuous evolutionary coefficient selection improves the auditable decision layer. The genetic search evolves the four charge/discharge scoring coefficients over a bounded continuous space. To keep the protocol reproducible, each zone uses deterministic initialization, 24 individuals, 14 generations, elitist tournament selection, Gaussian mutation, and a chronological training-plus-validation selection period. For long zones, selection uses a deterministic 720-day equidistant subset from the training-plus-validation period; all reported metrics are then evaluated on the complete final 20 percent held-out test split.

Table 11 reports the genetic soft-computing policy-search comparison on the final OPSD test split. The risk-adjusted genetic search improves downside stability relative to several simpler rules, while the result remains bounded by the stronger continuous-search and hindsight references.

|Method|Revenue|Regret|CVaR10|Loss days|Penalty|
|---|---|---|---|---|---|
|Genetic policy search (risk-adjusted)|21.653|11.817|-1.390|111|3.393|
|DF policy search (revenue)|21.639|11.831|-1.853|115|3.392|
|Genetic policy search (revenue)|21.626|11.844|-1.764|114|3.395|
|DF policy search (risk-adjusted)|21.487|11.983|-1.276|110|3.402|
|Robust quantile FTO|21.128|12.342|-0.585|97|3.295|
|Fuzzy risk-aware FTO|11.342|22.128|-4.968|259|3.295|
|Constrained Q-learning policy|-5.655|39.125|-24.534|1034|3.352|

![Figure 9. Genetic soft-computing policy search on the OPSD VPP test split, comparing revenue and risk-adjusted evolutionary objectives against transparent baselines and hindsight decision value.](figures/paper3_fig13_opsd_genetic_policy_search_vpp.png)

The genetic policy-search result is intentionally interpreted as a bounded improvement, not as a universal optimizer claim. The risk-adjusted genetic search reaches 21.653 EUR/day proxy revenue, slightly above the coarse revenue-selected decision-focused grid at 21.639 EUR/day, and improves CVaR10 relative to that grid from -1.853 to -1.390. The paired sign test against the revenue-selected grid is not significant because most daily schedules are identical or nearly identical under the two coefficient sets. The contribution is therefore methodological rather than promotional: a continuous soft-computing search can refine the auditable coefficient-selection layer, but the effect size is small and should be read together with robust quantile scheduling, fuzzy interpretability, and the Q-learning boundary check.

### 6.10 Multi-seed genetic-search stability

Table 12 reports the risk-adjusted genetic coefficient search with eight independent seeds under the same chronological selection and held-out test protocol. Across seeds, mean held-out revenue is 21.543 EUR/day proxy with standard deviation 0.078, and mean CVaR10 is -1.530 with standard deviation 0.163. Only 1/8 seeds matches or exceeds the coarse-grid revenue-selected policy in average revenue, but the seed-average CVaR10 remains less negative than the coarse-grid revenue-selected policy (-1.853). The result supports genetic search as a risk-stability refinement rather than an assured revenue maximization method.

|Metric|Value|Interpretation|
|---|---:|---|
|Seeds|8|Independent genetic-search runs under the same chronological protocol|
|Mean revenue across seeds|21.543 EUR/day|Stable but slightly below the single coarse-grid revenue baseline on average|
|Revenue standard deviation|0.078 EUR/day|Low seed sensitivity for the selected search design|
|Revenue range|21.425 to 21.648 EUR/day|One seed exceeds the coarse-grid revenue-selected policy|
|Mean CVaR10 across seeds|-1.530|Better downside profile than the revenue-selected coarse grid|
|CVaR10 range|-1.753 to -1.267|Risk performance varies, but stays near the intended risk-adjusted boundary|

![Figure 10. Multi-seed stability of genetic soft-computing policy search, summarizing held-out revenue and CVaR variability across independent seeds to separate reproducible refinement from single-run luck.](figures/paper3_fig15_opsd_genetic_multiseed_stability.png)

The multi-seed result strengthens the ASOC positioning because it turns the evolutionary component into an auditable stochastic-search experiment rather than a single decorative optimizer. At the same time, it prevents overclaiming: the best seed can match the coarse grid, but the average seed does not dominate it in revenue. The value of the genetic layer is therefore a bounded soft-computing refinement that explores continuous policy coefficients, improves downside behavior relative to the revenue-only coarse grid, and exposes the trade-off between mean revenue and CVaR under repeated random initialization.

### 6.11 Particle-swarm policy-search check

A separate particle-swarm optimization (PSO) check tests whether the continuous soft-computing result depends on the genetic operators. PSO searches the same four policy coefficients, uses the same chronological train-plus-validation selection period, and freezes the selected coefficients before the final held-out OPSD test split. The risk-adjusted PSO policy reaches 21.596 EUR/day proxy revenue, 11.874 regret, -1.741 CVaR10, and 113 loss days. The revenue-only PSO policy reaches 21.602 EUR/day proxy revenue and -1.967 CVaR10. Thus PSO is close to the decision-focused and genetic policies, but it does not exceed the risk-adjusted genetic search at 21.653 EUR/day proxy revenue and -1.390 CVaR10.

![Figure 11. Particle-swarm soft-computing policy search on the OPSD VPP test split, providing a second population-based continuous-search baseline for revenue, regret, and downside-risk comparison.](figures/paper3_fig16_opsd_pso_policy_search_vpp.png)

This result is useful precisely because it is not a promotional win. PSO confirms that a second swarm-intelligence optimizer can recover a competitive transparent policy in the same coefficient class, while the genetic search remains the strongest reported continuous-search variant. The paper therefore claims a reproducible soft-computing policy-search layer, not that any single metaheuristic universally dominates the decision grid.

### 6.12 Differentiable surrogate-policy baseline

To address whether a learning-style differentiable policy layer changes the conclusion, Table 13 adds a lightweight surrogate policy-gradient baseline. The surrogate uses the same chronological public OPSD split, but replaces discrete coefficient-grid selection with a differentiable soft-schedule objective. Charge and discharge probabilities are produced by softmax score functions over normalized rolling price, volatility, net-load shape, and intraday features. The soft objective approximates settlement revenue, imbalance penalty, charge-discharge overlap, and a revenue-plus-CVaR risk term. After CPU-only NumPy training on a deterministic 192-day selection subset per zone, the learned scoring weights are frozen and evaluated through the same hard four-hour charge/discharge simulator as the other policies.

Table 13 reports the differentiable surrogate-policy comparison on the final OPSD test split. The lightweight surrogate improves substantially over tabular Q-learning but remains below the strongest GA and PSO policies, so it is used as reviewer-response evidence rather than as a superiority claim.

|Method|Revenue|Regret|CVaR10|Loss days|Penalty|
|---|---:|---:|---:|---:|---:|
|Genetic policy search (risk-adjusted)|21.653|11.817|-1.390|111|3.393|
|PSO policy search (risk-adjusted)|21.596|11.874|-1.741|113|3.400|
|DF policy search (revenue)|21.639|11.831|-1.853|115|3.392|
|Surrogate policy-gradient (risk-adjusted)|21.265|12.205|-2.555|123|3.390|
|Surrogate policy-gradient (revenue)|20.939|12.531|-3.001|132|3.422|
|Fuzzy risk-aware FTO|11.342|22.128|-4.968|259|3.295|
|Constrained Q-learning policy|-5.655|39.125|-24.534|1034|3.352|

![Figure 12. Differentiable surrogate policy-gradient baseline on the OPSD VPP test split, showing how a lightweight gradient-updated policy compares with GA, PSO, Q-learning, and transparent decision baselines under the same chronological holdout.](figures/paper3_fig17_opsd_surrogate_policy_gradient_vpp.png)

The surrogate result is deliberately reported as a bounded reviewer-response experiment. The risk-adjusted surrogate reaches 21.265 EUR/day proxy revenue, which is below the genetic and PSO searches but above the fuzzy and constrained Q-learning baselines. Against the genetic risk-adjusted policy, the paired test gives 371 wins, 426 losses, and 581 ties, with a mean revenue delta of -0.331 EUR/day. This indicates that a lightweight differentiable surrogate is competitive enough to be informative, but it does not replace the stronger continuous soft-computing search in this public simulator. The conclusion therefore becomes stricter: learning-style policy evidence has been tested, and the current data support GA/PSO-style transparent coefficient search as the stronger auditable choice.

### 6.13 GIS infrastructure-context check

The decision simulator is intentionally public and compact, but the application setting is not small. A separate GIS evidence audit was therefore built from local copies of China power-transmission snapshots, OSM-derived mainland power-grid extracts, WRI China power-plant GeoJSON, and GEM China integrated-power tracker tables. The audit is not used to train the OPSD/UCI journal baselines and is not redistributed as raw data in the paper supplement. Its role is to test whether the VPP decision problem has a realistic resource-network motivation beyond a toy arbitrage example.

The 2025 grid snapshot contains 12,839 transmission-line records, 2,041 substation records, and 9,444 grid-link records. Across the available snapshots, the 2015-to-2025 transmission-line record count increases by 610.122%, while grid-link records increase by 114.588%, which supports the use of time-aware validation rather than a static infrastructure assumption. The OSM mainland extraction profiles 4.87 million power-grid records, including 216,700 line records and 194,613 generator points; renewable-or-storage source tags account for 98.832% of the generator-point table under the derived-summary denominator. WRI contributes 4,274 China power-plant records and a 25.407% renewable capacity share, while the GEM integrated tracker represents about 3,108.9 GW of operating capacity in the local China table and a 56.412% broad variable-or-dispatchable context share. Figure 13 summarizes the network-scale evidence and the high-voltage share visible in the transmission and substation layers. These statistics support the paper's framing of VPP bidding as a computer-science decision-learning problem under heterogeneous, dynamic, large-scale infrastructure context, while keeping the submitted numerical claims on the reproducible public OPSD simulator.

![Figure 13. Open GIS evidence for network-scale VPP decision context, summarizing China grid snapshot scale and high-voltage shares as external scenario evidence while excluding raw GIS files from the public supplement.](figures/paper3_fig14_china_grid_gis_externality.png)

"""
    text = text.replace(
        "\nThe decision results are deterministic chronological hold-out simulation metrics rather than inferential p-values.",
        "\n" + genetic_section + "The decision results are deterministic chronological hold-out simulation metrics rather than inferential p-values.",
    )
    genetic_appendix = """### A.6 Genetic soft-computing policy search

The genetic policy-search baseline uses the same four-coefficient scoring form as the decision-focused grid, but searches a bounded continuous coefficient space. The bounds are a_c, a_d in [-1.50, 1.50] and b_c, b_d in [-10.0, 10.0]. Each zone uses deterministic seeding, 24 individuals, 14 generations, five elites, tournament selection, convex parent mixing, Gaussian mutation, and occasional coordinate resampling. The revenue version maximizes mean selection-period revenue; the risk-adjusted version maximizes mean revenue plus 0.50 times CVaR10. For long zones, coefficient selection uses a deterministic 720-day equidistant subset of the chronological training-plus-validation period to keep the search reproducible and computationally bounded. Final reported metrics always use the complete final 20 percent held-out public test split.

The multi-seed stability check reruns only the risk-adjusted genetic search because it is the stochastic component most relevant to the paper's soft-computing claim. It uses eight independent seed offsets, the same population size, the same generation count, the same coefficient bounds, and the same chronological train/validation/test split. The output files `opsd_genetic_policy_multiseed_stability_by_seed.csv`, `opsd_genetic_policy_multiseed_stability_by_zone.csv`, `opsd_genetic_policy_multiseed_stability_coefficients.csv`, and `opsd_genetic_policy_multiseed_stability_summary.csv` document seed-level revenues, downside risk, coefficients, and summary statistics.

### A.7 Particle-swarm policy-search check

The particle-swarm baseline uses the same four-coefficient score policy as the decision-focused grid and genetic search, but replaces crossover/mutation with swarm velocity updates, personal bests, and a global best. Each zone uses 22 particles, 16 iterations, inertia 0.58, cognitive weight 1.28, social weight 1.42, the same coefficient bounds as the genetic search, and the same chronological selection subset cap of 720 train-plus-validation days. The PSO output files `opsd_pso_policy_search_vpp_daily.csv`, `opsd_pso_policy_search_vpp_summary.csv`, `opsd_pso_policy_search_vpp_test_aggregate.csv`, `opsd_pso_policy_search_vpp_coefficients.csv`, and `opsd_pso_policy_search_vpp_paired_tests.csv` document the full comparison. The intended interpretation is bounded: PSO is a competitive independent swarm check, not a new best policy over the genetic risk-adjusted variant.

### A.8 Differentiable surrogate-policy baseline

The surrogate policy-gradient baseline uses a CPU-only NumPy implementation and does not train an end-to-end price or load forecaster. For each day, normalized rolling price mean, rolling price volatility, rolling net-load shape, sine/cosine hour features, and a bias term feed separate charge and discharge score functions. A temperature-controlled softmax converts these scores into soft four-hour charge and discharge weights. The training objective approximates daily settlement value, imbalance penalty, charge-discharge overlap, and a revenue-plus-CVaR risk term. Numerical finite-difference gradients and an Adam-style update are run for 36 iterations on a deterministic 192-day selection subset per zone; the final hard-schedule evaluation still uses the same four-charge and four-discharge simulator as the other methods. Output files `opsd_surrogate_policy_gradient_vpp_daily.csv`, `opsd_surrogate_policy_gradient_vpp_summary.csv`, `opsd_surrogate_policy_gradient_vpp_test_aggregate.csv`, `opsd_surrogate_policy_gradient_vpp_coefficients.csv`, `opsd_surrogate_policy_gradient_vpp_training_trace.csv`, and `opsd_surrogate_policy_gradient_vpp_paired_tests.csv` document the result and training trace. This baseline tests a differentiable policy family without claiming neural-RL superiority or gradient propagation into the forecasting models.

"""
    text = text.replace("\n### A.6 Metrics and scripts", "\n" + genetic_appendix + "### A.9 Metrics and scripts")
    if "## Appendix A. Simulator and Reproducibility Details" not in text:
        appendix_block = """## Appendix A. Simulator and Reproducibility Details

### A.1 Public simulator information boundary

The simulator uses chronological OPSD price, load, wind, and solar variables to evaluate forecast-then-optimize and policy-search decisions under a declared information boundary. Policy coefficients are selected only on training and validation periods and are frozen before the final held-out test split. Realized prices and net-load values are used for settlement evaluation and hindsight upper-bound construction, not for selecting the implementable test policy.

### A.2 Resource and settlement abstraction

The public VPP abstraction includes a finite battery, flexible load shifting, charge and discharge limits, efficiency losses, transaction-cost proxies, and imbalance penalties. The simulator is intentionally compact so that the forecast-to-decision relationship can be audited. It does not claim to reproduce a full production market engine, network-constrained dispatch system, or confidential commercial settlement platform.

### A.3 Risk and regret metrics

Daily revenue is computed after applying the selected schedule to held-out settlement prices and net-load conditions. Regret is the difference between the hindsight daily value and the implementable policy value under the same public simulator. CVaR10 is the mean of the worst decile of daily revenues and is used as the downside-risk statistic for risk-aware selection.

### A.4 Forecast-coupled interface to Papers 1 and 2

The forecast-coupled experiment uses the price-forecasting and load-forecasting outputs from the preceding manuscripts as decision inputs. This appendix reports the interface at the level required for reproducibility: the forecasts are generated before held-out decision evaluation, then passed to the same VPP simulator used by the transparent baselines.

### A.5 Fuzzy risk-aware and constrained Q-learning baselines

The fuzzy comparator uses transparent price-level, volatility, and net-load-shape memberships. The constrained Q-learning comparator uses discretized state bins, remaining action counts, and chronological train/validation learning before frozen test evaluation. Both are reported to define the boundary of soft-computing and policy-learning evidence, not to claim universal superiority.

""" + genetic_appendix + """### A.9 Metrics and scripts

The reproducibility package includes public processed data, deterministic scripts, result CSV files, figure files, manifest hashes, and an audit report. The main scripts include `run_opsd_vpp_risk_simulator.py`, `run_opsd_decision_focused_policy_search.py`, `run_opsd_forecast_coupled_vpp.py`, `run_opsd_risk_aversion_sensitivity.py`, `run_opsd_vpp_reviewer_robustness.py`, `run_opsd_fuzzy_risk_vpp_baseline.py`, `run_opsd_constrained_q_learning_vpp_baseline.py`, `run_opsd_genetic_policy_search_vpp_baseline.py`, `run_opsd_genetic_policy_multiseed_stability.py`, `run_opsd_pso_policy_search_vpp_baseline.py`, `run_opsd_surrogate_policy_gradient_vpp_baseline.py`, `build_paper3_supplement_package.py`, and `verify_paper3_supplement_package.py`.

"""
        text = text.replace("\n## Data Availability", "\n" + appendix_block + "## Data Availability")
    text = text.replace(
        "`run_opsd_vpp_reviewer_robustness.py`, `run_opsd_fuzzy_risk_vpp_baseline.py`, and `run_opsd_constrained_q_learning_vpp_baseline.py`.",
        "`run_opsd_vpp_reviewer_robustness.py`, `run_opsd_fuzzy_risk_vpp_baseline.py`, `run_opsd_constrained_q_learning_vpp_baseline.py`, `run_opsd_genetic_policy_search_vpp_baseline.py`, `run_opsd_genetic_policy_multiseed_stability.py`, `run_opsd_pso_policy_search_vpp_baseline.py`, and `run_opsd_surrogate_policy_gradient_vpp_baseline.py`.",
    )
    text = re.sub(
        r"## Data Availability\n\n.*?(?=\n## References)",
        f"## Data Availability\n\n{DATA_STATEMENT}\n\n",
        text,
        flags=re.S,
    )

    declarations = f"""## CRediT Authorship Contribution Statement

{AUTHOR}: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review and editing, Visualization.

## Declaration of Competing Interest

The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Declaration of Generative AI and AI-Assisted Technologies in the Manuscript Preparation Process

{AI_DECLARATION}

"""
    text = text.replace("## References", declarations + "## References")

    front = f"""# {TITLE}

{AUTHOR}

{AFFILIATION}

Corresponding author: {AUTHOR}; {ADDRESS}; e-mail: {EMAIL}.

Article type: Technical paper / Research article.

Target journal: Applied Soft Computing.

"""
    return front + text + "\n"


def make_cover_letter() -> str:
    return f"""# Applied Soft Computing Paper 3 Cover Letter

Dear Editor,

I submit the manuscript "{TITLE}" for consideration as a technical research paper in Applied Soft Computing.

The manuscript fits the journal's soft-computing scope because it studies machine-learning, decision-support, time-series prediction, and power-and-energy applications through a real virtual-power-plant market-operation problem. The contribution is not a conventional electrical dispatch formulation. It is a decision-focused learning and predict-and-optimize framework that evaluates price and load forecasts by downstream market value, regret, CVaR, and loss-day risk.

The paper uses a fully reproducible public OPSD experimental layer. The public simulator includes battery capacity, flexible load shifting, imbalance penalties, chronological held-out evaluation, fuzzy risk-aware scoring, genetic soft-computing policy search with multi-seed stability diagnostics, particle-swarm policy search as an independent swarm-intelligence check, a lightweight surrogate policy-gradient baseline, constrained Q-learning as a policy-learning boundary check, and forecast-coupled policy tests that link the preceding price-forecasting and load-forecasting manuscripts to the virtual-power-plant decision layer. The forecast-coupled decision-focused policy obtains 31.43 EUR/day proxy revenue and 4.67 regret on the held-out coupled split, while the genetic search, PSO check, surrogate policy-gradient baseline, risk-aversion sweep, and Q-learning stress test transparently report the revenue-CVaR and policy-learning limits.

A public-data-only supplementary package has been prepared with processed public data, scripts, result CSV files, figures, verified references, requirements, manifest hashes, and an audit report. Local Hunan and Shandong operational records are not redistributed and are not required to reproduce the main claims.

The manuscript is original, has not been published previously, and is not under consideration elsewhere. The author declares no competing interests and no external funding. A generative-AI disclosure is included before the reference list, and no generative AI or AI-assisted tool was used to create or alter the manuscript figures or graphical abstract.

Sincerely,

{AUTHOR}

{AFFILIATION}

{COVER_ADDRESS}

{EMAIL}

ORCID: {ORCID}
"""


def make_declarations() -> str:
    return f"""# Applied Soft Computing Paper 3 Declarations

## Manuscript

{TITLE}

## Author and Correspondence

- Author: {AUTHOR}
- Affiliation: {AFFILIATION}
- Corresponding address: {ADDRESS}
- Email: {EMAIL}
- ORCID: {ORCID}

## Submission Declaration

The manuscript is original, has not been published previously except as an academic thesis or permissible preprint if later chosen by the author, is not under consideration elsewhere, and will not be published elsewhere in the same form without publisher permission if accepted.

## Declaration of Competing Interest

The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## CRediT Authorship Contribution Statement

{AUTHOR}: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review and editing, Visualization.

## Declaration of Generative AI and AI-Assisted Technologies

{AI_DECLARATION}

## Data Availability

{DATA_STATEMENT}

## Human Approval Gate

Before real upload, the author must confirm the author list, ORCID if requested by the portal, institutional policy on AI-use wording, and the final repository or supplementary-file route for the public reproducibility package.
"""


def make_data_code_pack() -> str:
    audit = SUPPLEMENT_AUDIT.read_text(encoding="utf-8") if SUPPLEMENT_AUDIT.exists() else ""
    audit = sanitize_upload_visible_audit_text(audit)
    supplement_zip = root_relative(SUPPLEMENT)
    supplement_audit = root_relative(SUPPLEMENT_AUDIT)
    supplement_dir = supplement_zip.removesuffix(".zip")
    return f"""# Applied Soft Computing Paper 3 Data and Code Availability Pack

## Public Dataset

The reproducible experimental layer uses the Open Power System Data time-series package, version 2020-10-06:

https://data.open-power-system-data.org/time_series/2020-10-06/

## Supplementary Package

- Package directory: `{supplement_dir}/`
- Zip archive: `{supplement_zip}`
- Audit report: `{supplement_audit}`
- Public GitHub repository: {GITHUB_REPO}
- Public GitHub release: {GITHUB_RELEASE}
- Release asset: {GITHUB_ASSET}
- Commit traceability: see the GitHub release receipt and repository history for the exact commit state used by the current pre-DOI release asset.
- DOI status: {DOI_STATUS}

## Included Reproducibility Assets

- Processed public OPSD data.
- Daily and aggregate result CSV files.
- Figure PNG files used by the manuscript.
- Verified reference assets and BibTeX.
- Requirements file and reproduction README.
- Scripts for VPP risk simulation, decision-focused policy search, genetic soft-computing policy search, multi-seed genetic stability checking, particle-swarm policy search, surrogate policy-gradient baseline testing, forecast-coupled VPP evaluation, risk-aversion sensitivity, reviewer robustness checks, constrained Q-learning boundary testing, and package verification.

## Recommended Data Statement for the Submission Portal

{DATA_STATEMENT}

## Private-Data Boundary

Local Hunan and Shandong operational records are not included in the public supplement. They are not necessary to reproduce the main paper claims, and they should remain local unless the author obtains explicit redistribution permission.

## Current Audit Summary

```text
{audit.strip()}
```
"""


def make_highlights() -> str:
    rows = ["# Applied Soft Computing Paper 3 Highlights", ""]
    for item in HIGHLIGHTS:
        rows.append(f"- {item} ({char_count(item)} characters)")
    rows.append("")
    rows.append("All highlights are within the 85-character limit including spaces.")
    return "\n".join(rows) + "\n"


def make_checklist(manuscript_text: str) -> str:
    wc = abstract_word_count(manuscript_text)
    high_status = "PASS" if 3 <= len(HIGHLIGHTS) <= 5 and all(char_count(h) <= 85 for h in HIGHLIGHTS) else "REVIEW"
    supp_exists = SUPPLEMENT.exists()
    return f"""# Applied Soft Computing Paper 3 Final Upload Checklist

Generated: 2026-06-30

Target journal: Applied Soft Computing.

Official sources:

- Aims and scope: {ASOC_SCOPE}
- Guide for authors: {ASOC_GUIDE}

## Files Built in This Pack

- paper_3_asoc_manuscript_candidate.docx
- paper_3_asoc_cover_letter.docx
- paper_3_asoc_declarations.docx
- paper_3_asoc_data_code_availability_pack.docx
- paper_3_asoc_highlights.docx
- paper_3_asoc_final_upload_checklist.docx
- paper_3_asoc_graphical_abstract.png
- paper_3_asoc_submission_metadata.json
- asoc_school_classification_confirmation_packet_2026-06-30.docx, if generated separately

## Compliance Check

| Gate | Current status | Required action before real upload |
|---|---|---|
| Scope | Machine/deep learning, decision support, power and energy, time-series prediction | Confirm school recognizes exact journal title/classification |
| Article type | Technical paper / research article | Select the matching Editorial Manager type |
| Abstract | {wc} words | Must stay at or below 250 words |
| Keywords | 7 keywords | Check portal accepts all seven |
| Highlights | {high_status}; 4 bullets, all <=85 characters | Upload separate editable highlights file |
| Graphical abstract | PNG generated from author data/diagrams, not generative AI | Upload separate graphical abstract file |
| Tables | Editable manuscript tables | Review final Word table layout after portal proof generation |
| Figures | Separate PNG assets exist | Upload figure files if the portal requests separate artwork |
| CRediT | Single-author CRediT statement drafted | Author approval required |
| Declaration of interest | "I have nothing to declare" drafted | Complete Elsevier declarations tool if required |
| Funding | No external funding statement drafted | Keep consistent in portal field |
| Generative AI declaration | Included before references | Author/institutional approval required |
| Data statement | Public supplement plus private-data boundary drafted | Deposit repository DOI/URL if available before acceptance |
| Supplementary package | {'Present' if supp_exists else 'Missing'} | Upload zip as supplementary material or deposit externally |
| School classification packet | {'Present' if SCHOOL_PACKET.exists() else 'Generate with work/build_paper3_asoc_school_confirmation_packet.py'} | Obtain written/screenshot confirmation for Applied Soft Computing before APC payment or formal reliance |
| Biography/photo | Not required in guide section inspected for this journal | Add only if portal requests it |

## Remaining Human Gates

- Obtain exact school-recognition confirmation for Applied Soft Computing under the current graduation-rule year before APC payment or formal reliance on this route.
- Decide whether to submit the public supplement as portal supplementary material first, deposit to Zenodo/OSF first, or do both.
- Author must confirm originality, no concurrent submission, final author list, AI-use disclosure wording, and private-data boundary.
"""


def make_graphical_abstract():
    width, height = 1328, 531
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def font(size: int, bold: bool = False):
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
        return ImageFont.load_default()

    title_font = font(30, bold=True)
    box_font = font(25, bold=True)
    metric_font = font(29, bold=True)
    body_font = font(24)
    note_font = font(16)

    def centered_multiline(x0, y0, x1, y1, label, fill, used_font):
        lines = label.split("\n")
        line_heights = [draw.textbbox((0, 0), line, font=used_font)[3] for line in lines]
        total_h = sum(line_heights) + (len(lines) - 1) * 8
        y = y0 + ((y1 - y0) - total_h) / 2
        for line, line_h in zip(lines, line_heights):
            bbox = draw.textbbox((0, 0), line, font=used_font)
            x = x0 + ((x1 - x0) - (bbox[2] - bbox[0])) / 2
            draw.text((x, y), line, fill=fill, font=used_font)
            y += line_h + 8

    draw.text((52, 34), "Decision-focused VPP bidding under price and load uncertainty", fill="#111827", font=title_font)
    boxes = [
        (52, 150, 270, 295, "Public forecasts\nprice + net load\nOPSD data"),
        (355, 150, 575, 295, "Decision-focused\npolicy search\nheld-out split"),
        (660, 150, 880, 295, "VPP simulator\nbattery + flex load\nimbalance penalty"),
        (965, 150, 1185, 295, "Market value\nrevenue, regret\nCVaR risk"),
    ]
    colors = ["#D9EAF7", "#E9E4F5", "#DDEFE6", "#F7E3D4"]
    for (x0, y0, x1, y1, label), color in zip(boxes, colors):
        draw.rounded_rectangle((x0, y0, x1, y1), radius=0, fill=color, outline="#334155", width=3)
        centered_multiline(x0, y0, x1, y1, label, "#1f2937", box_font)
    for start, end in [((282, 222), (342, 222)), ((587, 222), (647, 222)), ((892, 222), (952, 222))]:
        draw.line((start, end), fill="#334155", width=5)
        draw.polygon([(end[0], end[1]), (end[0] - 18, end[1] - 11), (end[0] - 18, end[1] + 11)], fill="#334155")
    draw.text((52, 350), "Forecast-coupled DF policy: 31.43 EUR/day proxy revenue", fill="#0f766e", font=metric_font)
    draw.text((52, 407), "Held-out regret: 4.67 | risk sweep exposes revenue-CVaR trade-offs", fill="#334155", font=body_font)
    draw.text(
        (52, 475),
        "Graphical abstract generated by a deterministic plotting script from manuscript concepts and public results; no generative AI artwork used.",
        fill="#64748b",
        font=note_font,
    )
    image.save(GRAPHICAL_ABSTRACT)


def write_doc(name: str, text: str):
    md = OUT / name.replace(".docx", ".md")
    md.write_text(text, encoding="utf-8")
    docx_text = re.sub(r"(?m)^```[A-Za-z0-9_-]*\s*$", "", text)
    docx_text = re.sub(r"(?m)^(\d+)\. ", "- ", docx_text).replace("`", "")
    rebuild_docx(md, docx_text)
    return md.with_suffix(".docx")


def main():
    manuscript = normalize_for_asoc()
    manuscript_path = write_doc("paper_3_asoc_manuscript_candidate.docx", manuscript)
    outputs = [
        manuscript_path,
        write_doc("paper_3_asoc_cover_letter.docx", make_cover_letter()),
        write_doc("paper_3_asoc_declarations.docx", make_declarations()),
        write_doc("paper_3_asoc_data_code_availability_pack.docx", make_data_code_pack()),
        write_doc("paper_3_asoc_highlights.docx", make_highlights()),
        write_doc("paper_3_asoc_final_upload_checklist.docx", make_checklist(manuscript)),
    ]
    make_graphical_abstract()
    if SUPPLEMENT.exists():
        shutil.copy2(SUPPLEMENT, OUT / SUPPLEMENT.name)
    abstract, keywords = extract_abstract_and_keywords(manuscript)
    metadata = {
        "target_journal": "Applied Soft Computing",
        "issn": "1568-4946",
        "article_type": "Technical paper / Research article",
        "title": TITLE,
        "author": AUTHOR,
        "affiliation": AFFILIATION,
        "email": EMAIL,
        "official_sources": {"aims_scope": ASOC_SCOPE, "guide_for_authors": ASOC_GUIDE},
        "abstract": abstract,
        "abstract_word_count": abstract_word_count(manuscript),
        "keywords": keywords,
        "highlights": [{"text": h, "characters": char_count(h)} for h in HIGHLIGHTS],
        "graphical_abstract": str(GRAPHICAL_ABSTRACT),
        "supplement_zip_copied": SUPPLEMENT.exists(),
        "human_gates": [
            "school classification confirmation for the exact journal title",
            "repository DOI/URL or supplementary-material route",
            "author approval of AI declaration and private-data boundary",
        ],
    }
    meta_path = OUT / "paper_3_asoc_submission_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in outputs + [GRAPHICAL_ABSTRACT, meta_path]:
        print(p)


if __name__ == "__main__":
    main()
