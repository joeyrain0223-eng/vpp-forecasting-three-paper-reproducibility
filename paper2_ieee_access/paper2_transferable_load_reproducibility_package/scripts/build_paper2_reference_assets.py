from __future__ import annotations

import csv
import html
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
AUDIT_OUT = ROOT / "references"
ASSET_OUT = ROOT / "references"
AUDIT_OUT.mkdir(parents=True, exist_ok=True)
ASSET_OUT.mkdir(parents=True, exist_ok=True)

CACHE = AUDIT_OUT / "paper2_crossref_cache.json"


REFERENCES = [
    {
        "number": 1,
        "key": "vaswani2017attention",
        "status": "url_verified_no_doi",
        "entry_type": "inproceedings",
        "authors": "Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia",
        "title": "Attention Is All You Need",
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
        "expected_title": "Temporal fusion transformers for interpretable multi-horizon time series forecasting",
    },
    {
        "number": 3,
        "key": "zhou2021informer",
        "doi": "10.1609/aaai.v35i12.17325",
        "expected_title": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting",
    },
    {
        "number": 4,
        "key": "wu2021autoformer",
        "status": "url_verified_no_doi",
        "entry_type": "inproceedings",
        "authors": "Wu, Haixu and Xu, Jiehui and Wang, Jianmin and Long, Mingsheng",
        "title": "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting",
        "booktitle": "Advances in Neural Information Processing Systems",
        "volume": "34",
        "pages": "22419--22430",
        "year": "2021",
        "url": "https://arxiv.org/abs/2106.13008",
        "note": "arXiv/NeurIPS metadata; no DOI inserted.",
    },
    {
        "number": 5,
        "key": "nie2023patchtst",
        "status": "url_verified_no_doi",
        "entry_type": "inproceedings",
        "authors": "Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant",
        "title": "A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers",
        "booktitle": "International Conference on Learning Representations",
        "year": "2023",
        "url": "https://openreview.net/forum?id=Jbdc0vTOcol",
        "note": "OpenReview ICLR metadata; no DOI inserted.",
    },
    {
        "number": 6,
        "key": "zeng2023dlinear",
        "doi": "10.1609/aaai.v37i9.26317",
        "expected_title": "Are Transformers Effective for Time Series Forecasting?",
    },
    {
        "number": 7,
        "key": "oreshkin2020nbeats",
        "status": "url_verified_no_doi",
        "entry_type": "inproceedings",
        "authors": "Oreshkin, Boris N. and Carpov, Dmitri and Chapados, Nicolas and Bengio, Yoshua",
        "title": "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting",
        "booktitle": "International Conference on Learning Representations",
        "year": "2020",
        "url": "https://openreview.net/forum?id=r1ecqn4YwB",
        "note": "OpenReview ICLR metadata; no DOI inserted.",
    },
    {
        "number": 8,
        "key": "bai2018tcn",
        "status": "url_verified_no_doi",
        "entry_type": "misc",
        "authors": "Bai, Shaojie and Kolter, J. Zico and Koltun, Vladlen",
        "title": "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling",
        "year": "2018",
        "url": "https://arxiv.org/abs/1803.01271",
        "note": "arXiv metadata; no DOI inserted.",
    },
    {
        "number": 9,
        "key": "dempster2020rocket",
        "doi": "10.1007/s10618-020-00701-z",
        "expected_title": "ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels",
    },
    {
        "number": 10,
        "key": "dempster2021minirocket",
        "doi": "10.1145/3447548.3467231",
        "expected_title": "MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification",
    },
    {
        "number": 11,
        "key": "yue2022ts2vec",
        "doi": "10.1609/aaai.v36i8.20881",
        "expected_title": "TS2Vec: Towards Universal Representation of Time Series",
    },
    {
        "number": 12,
        "key": "pan2010transfer",
        "doi": "10.1109/TKDE.2009.191",
        "expected_title": "A Survey on Transfer Learning",
    },
    {
        "number": 13,
        "key": "ganin2016domain",
        "status": "url_verified_no_doi",
        "entry_type": "article",
        "authors": "Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle, Hugo and Laviolette, François and Marchand, Mario and Lempitsky, Victor",
        "title": "Domain-Adversarial Training of Neural Networks",
        "journal": "Journal of Machine Learning Research",
        "volume": "17",
        "pages": "1--35",
        "year": "2016",
        "url": "https://jmlr.org/papers/v17/15-239.html",
        "note": "JMLR metadata; no DOI inserted.",
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
        "number": 17,
        "key": "uci2015electricity",
        "status": "url_verified_no_doi",
        "entry_type": "misc",
        "authors": "Trindade, Artur",
        "title": "ElectricityLoadDiagrams20112014",
        "howpublished": "UCI Machine Learning Repository",
        "year": "2015",
        "url": "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
        "note": "Public benchmark data source used for the multi-client transfer experiment.",
    },
    {
        "number": 18,
        "key": "opsd2020timeseries",
        "status": "url_verified_no_doi",
        "entry_type": "misc",
        "authors": "{Open Power System Data}",
        "title": "Time series data package",
        "howpublished": "Open Power System Data",
        "year": "2020",
        "url": "https://data.open-power-system-data.org/time_series/2020-10-06/",
        "note": "Public hourly load, price, wind, and solar time series used for the OPSD benchmark.",
    },
    {
        "number": 19,
        "key": "lago2021benchmark",
        "doi": "10.1016/j.apenergy.2021.116983",
        "expected_title": "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark",
    },
    {
        "number": 20,
        "key": "angelopoulos2023conformal",
        "doi": "10.1561/2200000101",
        "expected_title": "Conformal Prediction: A Gentle Introduction",
    },
    {
        "number": 21,
        "key": "uci2017appliances",
        "status": "url_verified_no_doi",
        "entry_type": "misc",
        "authors": "Candanedo, Luis M. and Feldheim, Véronique and Deramaix, Dominique",
        "title": "Appliances Energy Prediction",
        "howpublished": "UCI Machine Learning Repository",
        "year": "2017",
        "url": "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",
        "note": "Public residential appliance-energy benchmark used as an external one-hour-ahead load forecasting sanity check.",
    },
    {
        "number": 22,
        "key": "benidis2022deep",
        "doi": "10.1145/3533382",
        "expected_title": "Deep Learning for Time Series Forecasting: Tutorial and Literature Survey",
    },
    {
        "number": 23,
        "key": "hong2020energyforecasting",
        "doi": "10.1109/OAJPE.2020.3029979",
        "expected_title": "Energy Forecasting: A Review and Outlook",
    },
    {
        "number": 24,
        "key": "wang2019multiscale",
        "doi": "10.1016/j.energy.2019.03.080",
        "expected_title": "Deep learning for multi-scale smart energy forecasting",
    },
    {
        "number": 25,
        "key": "long2019dan",
        "doi": "10.1109/TPAMI.2018.2868685",
        "expected_title": "Transferable Representation Learning with Deep Adaptation Networks",
    },
    {
        "number": 26,
        "key": "liu2022scinet",
        "doi": "10.52202/068431-0421",
        "expected_title": "SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction",
    },
    {
        "number": 27,
        "key": "salinas2020deepar",
        "doi": "10.1016/j.ijforecast.2019.07.001",
        "expected_title": "DeepAR: Probabilistic forecasting with autoregressive recurrent networks",
    },
    {
        "number": 28,
        "key": "torres2021survey",
        "doi": "10.1089/big.2020.0159",
        "expected_title": "Deep Learning for Time Series Forecasting: A Survey",
    },
    {
        "number": 29,
        "key": "hewamalage2021rnn",
        "doi": "10.1016/j.ijforecast.2020.06.008",
        "expected_title": "Recurrent Neural Networks for Time Series Forecasting: Current status and future directions",
    },
    {
        "number": 30,
        "key": "petropoulos2022forecasting",
        "doi": "10.1016/j.ijforecast.2021.11.001",
        "expected_title": "Forecasting: theory and practice",
    },
]


def normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def crossref_by_doi(doi: str, cache: dict) -> dict:
    if doi in cache:
        return cache[doi]
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": "paper-reference-audit/1.0 (mailto:471062741@qq.com)"})
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=30, context=context) as fh:
        data = json.loads(fh.read().decode("utf-8"))["message"]
    cache[doi] = data
    time.sleep(0.2)
    return data


def authors_from_crossref(msg: dict) -> str:
    authors = []
    for a in msg.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        name = " ".join(x for x in [given, family] if x).strip()
        if name:
            authors.append(name)
    return html.unescape(", ".join(authors))


def display_authors(authors: str) -> str:
    return html.unescape(re.sub(r"\s+and\s+", ", ", authors or ""))


def issued_year(msg: dict) -> str:
    parts = msg.get("issued", {}).get("date-parts", [[]])
    return str(parts[0][0]) if parts and parts[0] else ""


def pages(msg: dict) -> str:
    return msg.get("page", "")


def doi_url(doi: str) -> str:
    return "https://doi.org/" + doi


def bibtex_escape(value: str) -> str:
    return html.unescape(str(value)).replace("{", "\\{").replace("}", "\\}")


def bibtex_entry(ref: dict, resolved: dict | None = None) -> str:
    entry_type = ref.get("entry_type", "article")
    key = ref["key"]
    fields = {}
    if resolved is not None:
        msg = resolved
        fields["author"] = authors_from_crossref(msg)
        fields["title"] = html.unescape((msg.get("title") or [ref.get("expected_title", "")])[0])
        if msg.get("container-title"):
            fields["journal"] = html.unescape(msg["container-title"][0])
        if msg.get("volume"):
            fields["volume"] = msg["volume"]
        if msg.get("issue"):
            fields["number"] = msg["issue"]
        if pages(msg):
            fields["pages"] = pages(msg)
        fields["year"] = issued_year(msg)
        fields["doi"] = ref["doi"]
        fields["url"] = doi_url(ref["doi"])
    else:
        for name in [
            "authors",
            "title",
            "booktitle",
            "journal",
            "series",
            "volume",
            "number",
            "pages",
            "year",
            "publisher",
            "address",
            "edition",
            "url",
            "note",
            "howpublished",
        ]:
            if ref.get(name):
                fields["author" if name == "authors" else name] = ref[name]
    lines = [f"@{entry_type}{{{key},"]
    for k, v in fields.items():
        lines.append(f"  {k} = {{{bibtex_escape(v)}}},")
    lines.append("}")
    return "\n".join(lines)


def markdown_reference(ref: dict, resolved: dict | None = None) -> str:
    n = ref["number"]
    if resolved is not None:
        msg = resolved
        authors = authors_from_crossref(msg)
        title = html.unescape((msg.get("title") or [ref.get("expected_title", "")])[0])
        venue = html.unescape((msg.get("container-title") or [""])[0])
        year = issued_year(msg)
        vol = msg.get("volume")
        issue = msg.get("issue")
        page = pages(msg)
        bits = [authors, f'"{title}"', venue]
        if vol:
            bits.append(f"vol. {vol}")
        if issue:
            bits.append(f"no. {issue}")
        if page:
            bits.append(f"pp. {page}")
        bits.append(year)
        bits.append(doi_url(ref["doi"]))
        return f"[{n}] " + ", ".join(x for x in bits if x) + "."
    authors = display_authors(ref.get("authors", ""))
    title = html.unescape(ref.get("title", ""))
    venue = html.unescape(ref.get("booktitle") or ref.get("journal") or ref.get("howpublished") or ref.get("publisher", ""))
    year = ref.get("year", "")
    url = ref.get("url", "")
    if not venue:
        return f'[{n}] {authors}, "{title}," {year}. {url}.'
    return f'[{n}] {authors}, "{title}," {venue}, {year}. {url}.'


def main() -> None:
    cache = load_cache()
    register_rows = []
    md_refs = []
    bib_entries = []
    for ref in REFERENCES:
        resolved = None
        status = ref.get("status", "doi_verified")
        title_match = ""
        resolved_title = ""
        if ref.get("doi"):
            resolved = crossref_by_doi(ref["doi"], cache)
            resolved_title = (resolved.get("title") or [""])[0]
            expected = normalize_title(ref.get("expected_title", ""))
            actual = normalize_title(resolved_title)
            title_match = "yes" if expected and (expected in actual or actual in expected) else "review"
            if title_match == "review":
                status = "doi_verified_title_review"
        register_rows.append(
            {
                "number": ref["number"],
                "key": ref["key"],
                "status": status,
                "doi": ref.get("doi", ""),
                "url": doi_url(ref["doi"]) if ref.get("doi") else ref.get("url", ""),
                "expected_title": ref.get("expected_title") or ref.get("title", ""),
                "resolved_title": resolved_title,
                "title_match": title_match,
                "note": ref.get("note", ""),
            }
        )
        md_refs.append(markdown_reference(ref, resolved))
        bib_entries.append(bibtex_entry(ref, resolved))

    save_cache(cache)
    csv_path = AUDIT_OUT / "paper2_verified_reference_register.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(register_rows[0].keys()))
        writer.writeheader()
        writer.writerows(register_rows)

    md_register = AUDIT_OUT / "paper2_verified_reference_register.md"
    md_register.write_text(
        "# Paper 2 Verified Reference Register\n\n"
        + "\n".join(
            f"- [{r['number']}] `{r['key']}` | {r['status']} | {r['url']} | title_match={r['title_match'] or 'n/a'}"
            for r in register_rows
        )
        + "\n",
        encoding="utf-8",
    )
    block = ASSET_OUT / "paper2_verified_reference_block.md"
    block.write_text("## References\n\n" + "\n".join(md_refs) + "\n", encoding="utf-8")
    bib = ASSET_OUT / "paper2_verified_references.bib"
    bib.write_text("\n\n".join(bib_entries) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_register)
    print(block)
    print(bib)


if __name__ == "__main__":
    main()
