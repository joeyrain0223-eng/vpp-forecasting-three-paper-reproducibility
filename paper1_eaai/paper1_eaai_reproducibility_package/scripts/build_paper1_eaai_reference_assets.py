from __future__ import annotations

import csv
import html
import json
import re
import textwrap
import time
import urllib.request
from urllib.error import HTTPError
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
OUT = ROOT / "references"
ASSET_OUT = ROOT / "references"
OUT.mkdir(parents=True, exist_ok=True)
ASSET_OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "crossref_verified_reference_cache.json"


REFERENCES = [
    {
        "number": 1,
        "key": "vaswani2017attention",
        "status": "url_verified_no_doi",
        "entry_type": "inproceedings",
        "authors": "Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia",
        "title": "Attention is all you need",
        "booktitle": "Advances in Neural Information Processing Systems",
        "volume": "30",
        "year": "2017",
        "url": "https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need",
        "note": "NeurIPS proceedings item; no DOI inserted.",
    },
    {
        "number": 2,
        "key": "lim2021temporal",
        "doi": "10.1016/j.ijforecast.2021.03.012",
        "expected_title": "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting",
    },
    {
        "number": 3,
        "key": "zhou2021informer",
        "doi": "10.1609/aaai.v35i12.17325",
        "expected_title": "Informer: Beyond efficient Transformer for long sequence time-series forecasting",
    },
    {
        "number": 4,
        "key": "wu2021autoformer",
        "status": "url_verified_no_crossref_doi",
        "entry_type": "inproceedings",
        "authors": "Wu, Haixu and Xu, Jiehui and Wang, Jianmin and Long, Mingsheng",
        "title": "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting",
        "booktitle": "Advances in Neural Information Processing Systems",
        "volume": "34",
        "pages": "22419--22430",
        "year": "2021",
        "url": "https://arxiv.org/abs/2106.13008",
        "note": "Crossref title search returned false matches; use arXiv/NeurIPS URL rather than an unverified DOI.",
    },
    {
        "number": 5,
        "key": "nie2023patchtst",
        "status": "url_verified_no_doi",
        "entry_type": "inproceedings",
        "authors": "Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant",
        "title": "A time series is worth 64 words: Long-term forecasting with Transformers",
        "booktitle": "International Conference on Learning Representations",
        "year": "2023",
        "url": "https://openreview.net/forum?id=Jbdc0vTOcol",
        "note": "OpenReview ICLR metadata; no DOI inserted.",
    },
    {
        "number": 6,
        "key": "zeng2023transformers",
        "doi": "10.1609/aaai.v37i9.26317",
        "expected_title": "Are Transformers effective for time series forecasting?",
    },
    {
        "number": 7,
        "key": "oreshkin2020nbeats",
        "status": "url_verified_no_doi",
        "entry_type": "inproceedings",
        "authors": "Oreshkin, Boris N. and Carpov, Dmitri and Chapados, Nicolas and Bengio, Yoshua",
        "title": "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting",
        "booktitle": "International Conference on Learning Representations",
        "year": "2020",
        "url": "https://openreview.net/forum?id=r1ecqn4YwB",
        "note": "OpenReview ICLR metadata; no DOI inserted.",
    },
    {
        "number": 8,
        "key": "wu2019graphwavenet",
        "doi": "10.24963/ijcai.2019/264",
        "expected_title": "Graph WaveNet for Deep Spatial-Temporal Graph Modeling",
    },
    {
        "number": 9,
        "key": "angelopoulos2023conformal",
        "doi": "10.1561/2200000101",
        "expected_title": "Conformal Prediction: A Gentle Introduction",
    },
    {
        "number": 10,
        "key": "bertsimas2020predictive",
        "doi": "10.1287/mnsc.2018.3253",
        "expected_title": "From Predictive to Prescriptive Analytics",
    },
    {
        "number": 11,
        "key": "elmachtoub2022smart",
        "doi": "10.1287/mnsc.2020.3922",
        "expected_title": "Smart Predict, then Optimize",
    },
    {
        "number": 12,
        "key": "amos2017optnet",
        "status": "url_verified_no_doi",
        "entry_type": "inproceedings",
        "authors": "Amos, Brandon and Kolter, J. Zico",
        "title": "OptNet: Differentiable Optimization as a Layer in Neural Networks",
        "booktitle": "Proceedings of the 34th International Conference on Machine Learning",
        "series": "Proceedings of Machine Learning Research",
        "volume": "70",
        "pages": "136--145",
        "year": "2017",
        "publisher": "PMLR",
        "url": "https://proceedings.mlr.press/v70/amos17a.html",
        "note": "PMLR metadata; no DOI inserted.",
    },
    {
        "number": 13,
        "key": "donti2017task",
        "status": "url_verified_no_doi",
        "entry_type": "inproceedings",
        "authors": "Donti, Priya L. and Amos, Brandon and Kolter, J. Zico",
        "title": "Task-based End-to-end Model Learning in Stochastic Optimization",
        "booktitle": "Advances in Neural Information Processing Systems",
        "volume": "30",
        "year": "2017",
        "url": "https://proceedings.neurips.cc/paper/2017/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf",
        "note": "NeurIPS proceedings item; no DOI inserted.",
    },
    {
        "number": 14,
        "key": "hong2014gefcom2012",
        "doi": "10.1016/j.ijforecast.2013.07.001",
        "expected_title": "Global Energy Forecasting Competition 2012",
    },
    {
        "number": 15,
        "key": "hong2016probabilistic",
        "doi": "10.1016/j.ijforecast.2016.02.001",
        "expected_title": "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond",
    },
    {
        "number": 16,
        "key": "lago2018spot",
        "doi": "10.1016/j.apenergy.2018.02.069",
        "expected_title": "Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms",
    },
    {
        "number": 17,
        "key": "lago2021dayahead",
        "doi": "10.1016/j.apenergy.2021.116983",
        "expected_title": "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark",
    },
    {
        "number": 18,
        "key": "weron2014price",
        "doi": "10.1016/j.ijforecast.2014.08.008",
        "expected_title": "Electricity price forecasting: A review of the state-of-the-art with a look into the future",
    },
    {
        "number": 19,
        "key": "uniejewski2019seasonal",
        "doi": "10.1016/j.eneco.2018.02.007",
        "expected_title": "On the importance of the long-term seasonal component in day-ahead electricity price forecasting",
        "note": "Replaces an unverifiable transfer-learning reference whose title did not match the Energy Economics article number.",
    },
    {
        "number": 20,
        "key": "nowotarski2018probabilistic",
        "doi": "10.1016/j.rser.2017.05.234",
        "expected_title": "Recent advances in electricity price forecasting: A review of probabilistic forecasting",
    },
    {
        "number": 21,
        "key": "bentaieb2017coherent",
        "status": "url_verified_no_doi",
        "entry_type": "inproceedings",
        "authors": "Ben Taieb, Souhaib and Taylor, James W. and Hyndman, Rob J.",
        "title": "Coherent probabilistic forecasts for hierarchical time series",
        "booktitle": "Proceedings of the 34th International Conference on Machine Learning",
        "series": "Proceedings of Machine Learning Research",
        "volume": "70",
        "pages": "3348--3357",
        "year": "2017",
        "publisher": "PMLR",
        "url": "https://proceedings.mlr.press/v70/taieb17a.html",
        "note": "PMLR metadata; no DOI inserted.",
    },
    {
        "number": 22,
        "key": "hyndman2021fpp3",
        "status": "url_verified_no_doi",
        "entry_type": "book",
        "authors": "Hyndman, Rob J. and Athanasopoulos, George",
        "title": "Forecasting: Principles and Practice",
        "edition": "3",
        "publisher": "OTexts",
        "address": "Melbourne, Australia",
        "year": "2021",
        "url": "https://otexts.com/fpp3/",
        "note": "Online book metadata; no DOI inserted.",
    },
    {
        "number": 23,
        "key": "opsd2020timeseries",
        "status": "dataset_url_verified",
        "entry_type": "misc",
        "authors": "{Open Power System Data}",
        "title": "Data Package Time Series, version 2020-10-06",
        "year": "2020",
        "url": "https://data.open-power-system-data.org/time_series/2020-10-06/",
        "note": "Dataset landing page and version URL.",
    },
    {
        "number": 24,
        "key": "rouzbahani2021vpp",
        "doi": "10.1016/j.seta.2021.101370",
        "expected_title": "A review on virtual power plant for energy management",
        "note": "Replaces an unverifiable VPP review entry.",
    },
    {
        "number": 25,
        "key": "nosratabadi2017microgridvpp",
        "doi": "10.1016/j.rser.2016.09.025",
        "expected_title": "A comprehensive review on microgrid and virtual power plant concepts employed for distributed energy resources scheduling in power systems",
        "note": "Replaces an unverifiable VPP management review entry.",
    },
    {
        "number": 26,
        "key": "boyd2004convex",
        "doi": "10.1017/cbo9780511804441",
        "expected_title": "Convex Optimization",
    },
    {
        "number": 27,
        "key": "hong2020energyforecasting",
        "doi": "10.1109/oajpe.2020.3029979",
        "expected_title": "Energy Forecasting: A Review and Outlook",
    },
    {
        "number": 28,
        "key": "ziel2018highdimensional",
        "doi": "10.1016/j.eneco.2017.12.016",
        "expected_title": "Day-ahead electricity price forecasting with high-dimensional structures: Univariate vs. multivariate modeling frameworks",
    },
    {
        "number": 29,
        "key": "gneiting2007proper",
        "doi": "10.1198/016214506000001437",
        "expected_title": "Strictly Proper Scoring Rules, Prediction, and Estimation",
    },
    {
        "number": 30,
        "key": "koenker1978quantiles",
        "doi": "10.2307/1913643",
        "expected_title": "Regression Quantiles",
    },
    {
        "number": 31,
        "key": "diebold1995comparing",
        "doi": "10.1080/07350015.1995.10524599",
        "expected_title": "Comparing Predictive Accuracy",
    },
    {
        "number": 32,
        "key": "wu2020mtgnn",
        "doi": "10.1145/3394486.3403118",
        "expected_title": "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks",
    },
    {
        "number": 33,
        "key": "yu2018stgcn",
        "doi": "10.24963/ijcai.2018/505",
        "expected_title": "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting",
    },
    {
        "number": 34,
        "key": "benidis2022deepts",
        "doi": "10.1098/rsta.2020.0209",
        "expected_title": "Time-series forecasting with deep learning: a survey",
    },
    {
        "number": 35,
        "key": "torres2021deeplearning",
        "doi": "10.1089/big.2020.0159",
        "expected_title": "Deep Learning for Time Series Forecasting: A Survey",
    },
]


def crossref_work(doi: str, cache: dict) -> dict:
    doi_key = doi.lower()
    if doi_key in cache:
        return cache[doi_key]
    url = f"https://api.crossref.org/works/{doi.lower()}"
    req = urllib.request.Request(url, headers={"User-Agent": "paper-reference-audit/1.0 (mailto:example@example.com)"})
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            if attempt:
                time.sleep(2.5 * attempt)
            with urllib.request.urlopen(req, timeout=30) as handle:
                data = json.load(handle)["message"]
            cache[doi_key] = data
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(1.0)
            return data
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
        except Exception as exc:  # network noise; retry conservatively
            last_error = exc
    raise RuntimeError(f"Crossref lookup failed after retries for {doi}: {last_error}")


def first_year(item: dict) -> str:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        parts = (item.get(key) or {}).get("date-parts")
        if parts and parts[0]:
            return str(parts[0][0])
    return ""


def names_from_crossref(item: dict) -> str:
    names = []
    for author in item.get("author") or []:
        given = author.get("given", "")
        family = author.get("family", "")
        name = " ".join(x for x in [given, family] if x).strip()
        if name:
            names.append(name)
    return " and ".join(names)


def clean_meta(value: str) -> str:
    return html.unescape(str(value or "")).strip()


def normalize_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def title_similarity(expected: str, actual: str) -> float:
    a = set(normalize_title(expected).split())
    b = set(normalize_title(actual).split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def bibtex_escape(value: str) -> str:
    return value.replace("&", "\\&")


def bibtex_entry(ref: dict, metadata: dict | None) -> str:
    entry_type = ref.get("entry_type")
    fields = {}
    if metadata:
        title = clean_meta("; ".join(metadata.get("title") or []))
        container = clean_meta("; ".join(metadata.get("container-title") or []))
        entry_type = "article" if container else "book"
        fields = {
            "author": clean_meta(names_from_crossref(metadata)),
            "title": title,
            "journal": container,
            "year": first_year(metadata),
            "volume": metadata.get("volume", ""),
            "number": metadata.get("issue", ""),
            "pages": metadata.get("page") or metadata.get("article-number", ""),
            "doi": metadata.get("DOI", "").lower(),
            "url": metadata.get("URL", ""),
        }
        if entry_type == "book":
            fields["publisher"] = clean_meta("; ".join(metadata.get("publisher") or [metadata.get("publisher", "")]))
    else:
        fields = {
            "author": ref.get("authors", ""),
            "title": ref.get("title", ""),
            "booktitle": ref.get("booktitle", ""),
            "journal": ref.get("journal", ""),
            "series": ref.get("series", ""),
            "volume": ref.get("volume", ""),
            "number": ref.get("number_field", ""),
            "pages": ref.get("pages", ""),
            "publisher": ref.get("publisher", ""),
            "address": ref.get("address", ""),
            "edition": ref.get("edition", ""),
            "year": ref.get("year", ""),
            "url": ref.get("url", ""),
            "note": ref.get("note", ""),
        }
    ordered = [
        "author",
        "title",
        "journal",
        "booktitle",
        "series",
        "volume",
        "number",
        "pages",
        "publisher",
        "address",
        "edition",
        "year",
        "doi",
        "url",
        "note",
    ]
    lines = [f"@{entry_type}{{{ref['key']},"]
    for key in ordered:
        value = fields.get(key)
        if value:
            lines.append(f"  {key} = {{{bibtex_escape(str(value))}}},")
    lines.append("}")
    return "\n".join(lines)


def plain_reference(ref: dict, metadata: dict | None) -> str:
    if metadata:
        authors = metadata.get("author") or []
        author_txt = clean_meta(", ".join(
            " ".join(part for part in [a.get("given", ""), a.get("family", "")] if part).strip()
            for a in authors
        ))
        title = clean_meta("; ".join(metadata.get("title") or []))
        container = clean_meta("; ".join(metadata.get("container-title") or []))
        year = first_year(metadata)
        volume = metadata.get("volume", "")
        pages = metadata.get("page") or metadata.get("article-number", "")
        doi = metadata.get("DOI", "").lower()
        bits = [f"[{ref['number']}] {author_txt}, {title}"]
        if container:
            vol_pages = " ".join(x for x in [volume, f"({year})", pages] if x)
            bits.append(f"{container} {vol_pages}".strip())
        else:
            bits.append(f"({year})")
        bits.append(f"https://doi.org/{doi}.")
        return ", ".join(bits)

    authors = ref.get("authors", "").replace(" and ", ", ")
    title = ref.get("title", "")
    venue = ref.get("journal") or ref.get("booktitle") or ref.get("publisher", "")
    extra = " ".join(x for x in [ref.get("volume", ""), f"({ref.get('year', '')})", ref.get("pages", "")] if x).strip()
    url = ref.get("url", "")
    return f"[{ref['number']}] {authors}, {title}, {venue} {extra}, {url}.".replace("  ", " ")


def main() -> None:
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        cache = {}
    rows = []
    bib_entries = []
    plain_refs = []
    for ref in REFERENCES:
        metadata = None
        status = ref.get("status", "doi_confirmed")
        similarity = ""
        actual_title = ref.get("title", "")
        container = ""
        year = ref.get("year", "")
        doi = ref.get("doi", "")
        if doi:
            metadata = crossref_work(doi, cache)
            actual_title = clean_meta("; ".join(metadata.get("title") or []))
            container = clean_meta("; ".join(metadata.get("container-title") or []))
            year = first_year(metadata)
            similarity = f"{title_similarity(ref.get('expected_title', actual_title), actual_title):.3f}"
            if float(similarity) < 0.70:
                status = "doi_title_mismatch_review_required"
            else:
                status = "doi_confirmed"
        else:
            container = ref.get("booktitle") or ref.get("journal") or ref.get("publisher", "")
        rows.append(
            {
                "number": ref["number"],
                "key": ref["key"],
                "status": status,
                "doi": doi.lower(),
                "title": actual_title,
                "container": container,
                "year": year,
                "similarity": similarity,
                "url": (metadata or ref).get("URL") if metadata else ref.get("url", ""),
                "note": ref.get("note", ""),
            }
        )
        bib_entries.append(bibtex_entry(ref, metadata))
        plain_refs.append(plain_reference(ref, metadata))

    csv_path = OUT / "paper1_eaai_verified_reference_register.csv"
    md_path = OUT / "paper1_eaai_verified_reference_register.md"
    bib_path = ASSET_OUT / "paper1_eaai_verified_references.bib"
    refs_path = ASSET_OUT / "paper1_eaai_verified_reference_block.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bib_path.write_text("\n\n".join(bib_entries) + "\n", encoding="utf-8")
    refs_path.write_text("## References\n\n" + "\n".join(plain_refs) + "\n", encoding="utf-8")

    confirmed = sum(1 for row in rows if row["status"] == "doi_confirmed")
    no_doi = sum(
        1
        for row in rows
        if row["status"] in {"url_verified_no_doi", "url_verified_no_crossref_doi", "dataset_url_verified"}
    )
    review = len(rows) - confirmed - no_doi
    md = f"""# Paper 1 EAAI Verified Reference Register

Generated by: `work/build_paper1_eaai_reference_assets.py`

Summary:

- DOI-confirmed entries: {confirmed}
- URL/dataset entries without inserted DOI: {no_doi}
- Review-required entries: {review}

Important corrections made in this verified register:

- Reference 19 replaces an unverifiable transfer-learning citation with a DOI-confirmed day-ahead electricity-price forecasting paper by Uniejewski, Marcjasz, and Weron.
- References 24 and 25 replace unverifiable virtual-power-plant review entries with DOI-confirmed VPP review literature.
- References 27-35 add Crossref-confirmed depth for energy forecasting review, high-dimensional electricity-price modeling, proper probabilistic scoring, quantile regression, predictive-accuracy testing, graph time-series learning, and deep time-series forecasting surveys.
- NeurIPS, ICLR/OpenReview, PMLR, OTexts, and OPSD entries keep stable URLs instead of unverified automated DOI matches.

| # | Key | Status | DOI | Title | Container | Year | Note |
|---|---|---|---|---|---|---|---|
"""
    for row in rows:
        md += "| {number} | {key} | {status} | {doi} | {title} | {container} | {year} | {note} |\n".format(
            **{k: str(v).replace("|", "\\|") for k, v in row.items()}
        )
    md += textwrap.dedent(
        f"""

## Output Files

- BibTeX: `{bib_path}`
- Manuscript reference block: `{refs_path}`
- CSV register: `{csv_path}`

## Remaining Manual Checks

- Import the BibTeX into Zotero or EndNote before final submission to normalize capitalization and journal abbreviations.
- Recheck journal-required reference style after the target journal is fixed in the submission system.
- Keep the old automated Crossref audit as a false-positive warning; do not reinsert its rejected first hits.
"""
    )
    md_path.write_text(md, encoding="utf-8")
    print(csv_path)
    print(md_path)
    print(bib_path)
    print(refs_path)


if __name__ == "__main__":
    main()
