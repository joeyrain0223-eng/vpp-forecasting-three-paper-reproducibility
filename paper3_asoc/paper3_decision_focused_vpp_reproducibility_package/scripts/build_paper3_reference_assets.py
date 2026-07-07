from __future__ import annotations

import csv
import html
import json
import re
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

CACHE = AUDIT_OUT / "paper3_crossref_cache.json"


REFERENCES = [
    {
        "number": 1,
        "key": "bertsimas2020prescriptive",
        "doi": "10.1287/mnsc.2018.3253",
        "expected_title": "From Predictive to Prescriptive Analytics",
    },
    {
        "number": 2,
        "key": "elmachtoub2022smart",
        "doi": "10.1287/mnsc.2020.3922",
        "expected_title": "Smart Predict, then Optimize",
    },
    {
        "number": 3,
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
        "url": "https://proceedings.mlr.press/v70/amos17a.html",
        "note": "PMLR proceedings item; no DOI inserted.",
    },
    {
        "number": 4,
        "key": "donti2017task",
        "status": "url_verified_no_doi",
        "entry_type": "inproceedings",
        "authors": "Donti, Priya L. and Amos, Brandon and Kolter, J. Zico",
        "title": "Task-based End-to-end Model Learning in Stochastic Optimization",
        "booktitle": "Advances in Neural Information Processing Systems",
        "volume": "30",
        "year": "2017",
        "url": "https://papers.neurips.cc/paper/7132-task-based-end-to-end-model-learning-in-stochastic-optimization",
        "note": "NeurIPS proceedings item; no DOI inserted.",
    },
    {
        "number": 5,
        "key": "wilder2019melding",
        "doi": "10.1609/aaai.v33i01.33011658",
        "expected_title": "Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization",
    },
    {
        "number": 6,
        "key": "rockafellar2000cvar",
        "doi": "10.21314/JOR.2000.038",
        "expected_title": "Optimization of conditional value-at-risk",
    },
    {
        "number": 7,
        "key": "boyd2004convex",
        "doi": "10.1017/CBO9780511804441",
        "expected_title": "Convex Optimization",
        "entry_type": "book",
    },
    {
        "number": 8,
        "key": "angelopoulos2023conformal",
        "doi": "10.1561/2200000101",
        "expected_title": "Conformal Prediction: A Gentle Introduction",
    },
    {
        "number": 9,
        "key": "lim2021temporal",
        "doi": "10.1016/j.ijforecast.2021.03.012",
        "expected_title": "Temporal fusion transformers for interpretable multi-horizon time series forecasting",
    },
    {
        "number": 10,
        "key": "lago2021benchmark",
        "doi": "10.1016/j.apenergy.2021.116983",
        "expected_title": "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark",
    },
    {
        "number": 11,
        "key": "hong2016probabilistic",
        "doi": "10.1016/j.ijforecast.2016.02.001",
        "expected_title": "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond",
    },
    {
        "number": 12,
        "key": "rouzbahani2021vppreview",
        "doi": "10.1016/j.seta.2021.101370",
        "expected_title": "A review on virtual power plant for energy management",
    },
    {
        "number": 13,
        "key": "nosratabadi2017microgridvpp",
        "doi": "10.1016/j.rser.2016.09.025",
        "expected_title": "A comprehensive review on microgrid and virtual power plant concepts employed for distributed energy resources scheduling in power systems",
    },
    {
        "number": 14,
        "key": "nguyen2018vppbidding",
        "doi": "10.1109/TIA.2018.2828379",
        "expected_title": "A Bidding Strategy for Virtual Power Plants With the Intraday Demand Response Exchange Market Using the Stochastic Programming",
    },
    {
        "number": 15,
        "key": "shafiekhani2019strategic",
        "doi": "10.1016/j.ijepes.2019.05.023",
        "expected_title": "Strategic bidding of virtual power plant in energy markets: A bi-level multi-objective approach",
    },
    {
        "number": 16,
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
        "number": 17,
        "key": "zadeh1965fuzzy",
        "doi": "10.1016/S0019-9958(65)90241-X",
        "expected_title": "Fuzzy sets",
    },
    {
        "number": 18,
        "key": "takagi1985fuzzy",
        "doi": "10.1109/TSMC.1985.6313399",
        "expected_title": "Fuzzy identification of systems and its applications to modeling and control",
    },
    {
        "number": 19,
        "key": "jang1993anfis",
        "doi": "10.1109/21.256541",
        "expected_title": "ANFIS: adaptive-network-based fuzzy inference system",
    },
    {
        "number": 20,
        "key": "kar2014neurofuzzy",
        "doi": "10.1016/j.asoc.2013.10.014",
        "expected_title": "Applications of neuro fuzzy systems: A brief review and future outline",
    },
    {
        "number": 21,
        "key": "alawami2017fuzzyvpp",
        "doi": "10.1109/TIA.2017.2723338",
        "expected_title": "Optimal Demand Response Bidding and Pricing Mechanism With Fuzzy Optimization: Application for a Virtual Power Plant",
    },
    {
        "number": 22,
        "key": "vazquez2019rl",
        "doi": "10.1016/j.apenergy.2018.11.002",
        "expected_title": "Reinforcement learning for demand response: A review of algorithms and modeling techniques",
    },
    {
        "number": 23,
        "key": "weron2014epf",
        "doi": "10.1016/j.ijforecast.2014.08.008",
        "expected_title": "Electricity price forecasting: A review of the state-of-the-art with a look into the future",
    },
    {
        "number": 24,
        "key": "nowotarski2018probepf",
        "doi": "10.1016/j.rser.2017.05.234",
        "expected_title": "Recent advances in electricity price forecasting: A review of probabilistic forecasting",
    },
    {
        "number": 25,
        "key": "pandzic2013offering",
        "doi": "10.1016/j.apenergy.2012.12.077",
        "expected_title": "Offering model for a virtual power plant based on stochastic programming",
    },
    {
        "number": 26,
        "key": "mashhour2011part1",
        "doi": "10.1109/TPWRS.2010.2070884",
        "expected_title": "Bidding Strategy of Virtual Power Plant for Participating in Energy and Spinning Reserve Markets—Part I: Problem Formulation",
    },
    {
        "number": 27,
        "key": "dabbagh2016industrialvpp",
        "doi": "10.1016/j.apenergy.2015.12.024",
        "expected_title": "Stochastic profit-based scheduling of industrial virtual power plant using the best demand response strategy",
    },
    {
        "number": 28,
        "key": "yang2018bioinspired",
        "doi": "10.1016/j.asoc.2018.04.051",
        "expected_title": "Hybrid bio-Inspired computational intelligence techniques for solving power system optimization problems: A comprehensive survey",
    },
    {
        "number": 29,
        "key": "saber2021adaptivefuzzy",
        "doi": "10.1016/j.asoc.2020.106882",
        "expected_title": "Adaptive optimal fuzzy logic based energy management in multi-energy microgrid considering operational uncertainties",
    },
    {
        "number": 30,
        "key": "mamaghani2017fuzzymicrogrid",
        "doi": "10.1016/j.asoc.2017.05.059",
        "expected_title": "Hierarchical genetic optimization of a fuzzy logic system for energy flows management in microgrids",
    },
    {
        "number": 31,
        "key": "mandi2024dflsurvey",
        "doi": "10.1613/jair.1.15320",
        "expected_title": "Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities",
    },
    {
        "number": 32,
        "key": "li2024vppoperationsreview",
        "doi": "10.1016/j.apenergy.2023.122284",
        "expected_title": "Review of virtual power plant operations: Resource coordination and multidimensional interaction",
    },
    {
        "number": 33,
        "key": "du2023drlsmartgrid",
        "doi": "10.1109/JPROC.2023.3303358",
        "expected_title": "Deep Reinforcement Learning for Smart Grid Operations: Algorithms, Applications, and Prospects",
    },
    {
        "number": 34,
        "key": "bental2009robust",
        "doi": "10.1515/9781400831050",
        "expected_title": "Robust Optimization",
        "entry_type": "book",
    },
    {
        "number": 35,
        "key": "birge2011stochastic",
        "doi": "10.1007/978-1-4614-0237-4",
        "expected_title": "Introduction to Stochastic Programming",
        "entry_type": "book",
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
    req = urllib.request.Request(url, headers={"User-Agent": "paper-reference-audit/1.0 (mailto:none@example.com)"})
    with urllib.request.urlopen(req, timeout=30) as fh:
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
            container = "booktitle" if entry_type == "inproceedings" else "journal"
            fields[container] = html.unescape(msg["container-title"][0])
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
    csv_path = AUDIT_OUT / "paper3_verified_reference_register.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(register_rows[0].keys()))
        writer.writeheader()
        writer.writerows(register_rows)

    md_register = AUDIT_OUT / "paper3_verified_reference_register.md"
    md_register.write_text(
        "# Paper 3 Verified Reference Register\n\n"
        + "\n".join(
            f"- [{r['number']}] `{r['key']}` | {r['status']} | {r['url']} | title_match={r['title_match'] or 'n/a'}"
            for r in register_rows
        )
        + "\n",
        encoding="utf-8",
    )
    block = ASSET_OUT / "paper3_verified_reference_block.md"
    block.write_text("## References\n\n" + "\n".join(md_refs) + "\n", encoding="utf-8")
    bib = ASSET_OUT / "paper3_verified_references.bib"
    bib.write_text("\n\n".join(bib_entries) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_register)
    print(block)
    print(bib)


if __name__ == "__main__":
    main()
