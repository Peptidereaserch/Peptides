
# For you fools who may want to scrape later:
# Usage:
#   python pci_db_psms_all_columns.py --out psms.csv
#   python pci_db_psms_all_columns.py --out psms.csv --max-pages 10
#   python pci_db_psms_all_columns.py --out psms.csv --params "mhc_class=I&disease=melanoma"

from __future__ import annotations

import argparse
import csv
import random
import time
from typing import Dict, List, Optional
from urllib.parse import parse_qsl

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://pci-db.org/psms/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; pci-db-scraper/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


EXPECTED_COLUMNS = [
    "Peptide Sequence",
    "Best HLA Allele",
    "Tissue",
    "Disease",
    "MHC Class",
    "Peptide Modifications",
    "Uniprot IDs",
    "Affinity % Rank",
]


def clean_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def parse_params_string(s: str) -> Dict[str, str]:
    s = (s or "").strip()
    if not s:
        return {}
    return {k: v for k, v in parse_qsl(s, keep_blank_values=True)}


def find_psm_table(soup: BeautifulSoup):
    for tbl in soup.find_all("table"):
        ths = [clean_text(th.get_text(" ", strip=True)) for th in tbl.find_all("th")]
        if all(col in ths for col in EXPECTED_COLUMNS):
            return tbl
    return None


def extract_table_headers(tbl) -> List[str]:
    return [clean_text(th.get_text(" ", strip=True)) for th in tbl.find_all("th")]


def extract_rows(tbl, headers: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for tr in tbl.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        # Keep only as many cells as we have headers (ignore extra action/link columns if they exist)
        values = [clean_text(td.get_text(" ", strip=True)) for td in tds[: len(headers)]]

        # If a row is shorter than headers, pad with blanks
        if len(values) < len(headers):
            values += [""] * (len(headers) - len(values))

        row = dict(zip(headers, values))

        # Skip truly empty rows
        if any(v for v in row.values()):
            rows.append(row)

    return rows


def has_next_page(soup: BeautifulSoup) -> bool:
    # Looks for a pagination link labeled "Next"
    for a in soup.find_all("a"):
        if clean_text(a.get_text(" ", strip=True)).lower().startswith("next"):
            return True
    return False


def scrape(
    base_url: str,
    extra_params: Dict[str, str],
    start_page: int = 1,
    max_pages: Optional[int] = None,
    sleep_s: float = 1.0,
    jitter_s: float = 0.5,
    timeout: int = 30,
) -> (List[str], List[Dict[str, str]]):
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows: List[Dict[str, str]] = []
    headers: Optional[List[str]] = None

    page = start_page
    while True:
        if max_pages is not None and (page - start_page + 1) > max_pages:
            break

        params = dict(extra_params)
        params["page"] = str(page)

        resp = session.get(base_url, params=params, timeout=timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        tbl = find_psm_table(soup)
        if tbl is None:
            # If the site changes or there’s no table found, stop
            break

        if headers is None:
            headers = extract_table_headers(tbl)

            missing = [c for c in EXPECTED_COLUMNS if c not in headers]
            if missing:
                raise RuntimeError(f"Table headers changed; missing columns: {missing}")

        rows = extract_rows(tbl, headers=headers)
        if not rows:
            break

        all_rows.extend(rows)

        if not has_next_page(soup):
            break

        time.sleep(max(0.0, sleep_s + random.uniform(0.0, jitter_s)))
        page += 1

    if headers is None:
        headers = EXPECTED_COLUMNS[:]  
    return headers, all_rows


def write_csv(path: str, headers: List[str], rows: List[Dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--params", default="", help='Extra query params ')
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    args = ap.parse_args()

    extra = parse_params_string(args.params)
    headers, rows = scrape(
        base_url=args.base_url,
        extra_params=extra,
        start_page=args.start_page,
        max_pages=args.max_pages,
        sleep_s=args.sleep,
        jitter_s=args.jitter,
    )

    write_csv(args.out, headers, rows)
    print(f"Saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
