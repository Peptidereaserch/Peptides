# Again for you fools 
# Usage:
#   python SeccondScraper.py --out psms.csv
#   python SeccondScraper.py --out psms.csv --max-pages 10
#   python SeccondScraper.py --out psms.csv --params "mhc_class=I&disease=melanoma"
#  
from __future__ import annotations

import argparse
import csv
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
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

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_thread_local = threading.local()


def _get_session() -> requests.Session:
    """
    requests.Session is NOT thread-safe; we create one per thread via thread-local storage.
    """
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(HEADERS)
        _thread_local.session = sess
    return sess


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

        values = [clean_text(td.get_text(" ", strip=True)) for td in tds[: len(headers)]]

        if len(values) < len(headers):
            values += [""] * (len(headers) - len(values))

        row = dict(zip(headers, values))

        if any(v for v in row.values()):
            rows.append(row)

    return rows


def has_next_page(soup: BeautifulSoup) -> bool:
    for a in soup.find_all("a"):
        if clean_text(a.get_text(" ", strip=True)).lower().startswith("next"):
            return True
    return False


def _sleep_with_jitter(seconds: float, jitter: float) -> None:
    time.sleep(max(0.0, seconds + random.uniform(0.0, jitter)))


def request_with_retries(
    url: str,
    params: Dict[str, str],
    timeout: int,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
    jitter_s: float,
) -> requests.Response:

    session = _get_session()

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)

            # success
            if resp.status_code < 400:
                return resp

            # non-retryable HTTP error
            if resp.status_code not in RETRYABLE_STATUS:
                resp.raise_for_status()
                return resp  # unreachable

            # retryable HTTP error
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait_s = float(retry_after)
                except ValueError:
                    wait_s = backoff_base * (2 ** attempt)
            else:
                wait_s = backoff_base * (2 ** attempt)

            wait_s = min(wait_s, backoff_max)

            # If we're out of retries, raise
            if attempt >= max_retries:
                resp.raise_for_status()

            _sleep_with_jitter(wait_s, jitter_s)
            continue

        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = e
            if attempt >= max_retries:
                raise
            wait_s = min(backoff_base * (2 ** attempt), backoff_max)
            _sleep_with_jitter(wait_s, jitter_s)

    # should never hit
    if last_err:
        raise last_err
    raise RuntimeError("request_with_retries failed unexpectedly")


def fetch_page(
    base_url: str,
    extra_params: Dict[str, str],
    page: int,
    timeout: int,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
    jitter_s: float,
) -> Tuple[int, Optional[List[str]], List[Dict[str, str]], bool]:
    """
    Fetch and parse a single page.
    Returns: (page_number, headers_or_None, rows, has_next)
    """
    params = dict(extra_params)
    params["page"] = str(page)

    resp = request_with_retries(
        base_url, params=params, timeout=timeout,
        max_retries=max_retries, backoff_base=backoff_base,
        backoff_max=backoff_max, jitter_s=jitter_s
    )

    soup = BeautifulSoup(resp.text, "html.parser")
    tbl = find_psm_table(soup)
    if tbl is None:
        # treat as end / unexpected layout
        return page, None, [], False

    headers = extract_table_headers(tbl)

    missing = [c for c in EXPECTED_COLUMNS if c not in headers]
    if missing:
        raise RuntimeError(f"Table headers changed; missing columns: {missing}")

    rows = extract_rows(tbl, headers=headers)
    nxt = has_next_page(soup)

    return page, headers, rows, nxt


def scrape(
    base_url: str,
    extra_params: Dict[str, str],
    start_page: int = 1,
    max_pages: Optional[int] = None,
    workers: int = 4,
    timeout: int = 30,
    max_retries: int = 6,
    backoff_base: float = 0.75,
    backoff_max: float = 30.0,
    jitter_s: float = 0.25,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Concurrent, ordered pagination:
    - Fetches several pages at once (workers)
    - Processes results in page order
    - Stops when the first page with has_next=False is reached
    """
    all_rows: List[Dict[str, str]] = []
    headers: Optional[List[str]] = None

    # page we want to commit next (in-order)
    next_to_process = start_page

    # bound how many pages we will try, if max_pages is provided
    def within_limit(p: int) -> bool:
        if max_pages is None:
            return True
        return (p - start_page + 1) <= max_pages

    in_flight: Dict[int, "concurrent.futures.Future"] = {}
    stop_at_page: Optional[int] = None  # once known, we won't schedule beyond it

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        # seed initial window
        p = start_page
        while len(in_flight) < max(1, workers) and within_limit(p):
            in_flight[p] = ex.submit(
                fetch_page,
                base_url, extra_params, p,
                timeout, max_retries, backoff_base, backoff_max, jitter_s
            )
            p += 1

        results_buffer: Dict[int, Tuple[Optional[List[str]], List[Dict[str, str]], bool]] = {}

        while in_flight:
            # wait for any page to complete
            done_any = False
            for fut in as_completed(list(in_flight.values())):
                # identify which page finished
                finished_page = None
                for pg, f in list(in_flight.items()):
                    if f is fut:
                        finished_page = pg
                        del in_flight[pg]
                        break

                if finished_page is None:
                    continue

                page_num, page_headers, page_rows, page_has_next = fut.result()
                results_buffer[page_num] = (page_headers, page_rows, page_has_next)
                done_any = True

                # Try to process in-order pages as far as we can
                while next_to_process in results_buffer:
                    ph, pr, hn = results_buffer.pop(next_to_process)

                    # no table / empty rows: treat as end
                    if not pr and ph is None:
                        stop_at_page = next_to_process - 1
                        # clear anything beyond; we will break after draining in-flight
                        break

                    if headers is None and ph is not None:
                        headers = ph

                    # If headers still None (shouldn't happen unless site changed)
                    if headers is None:
                        headers = EXPECTED_COLUMNS[:]

                    all_rows.extend(pr)

                    # if this page says "no next", we stop here
                    if not hn:
                        stop_at_page = next_to_process
                        break

                    next_to_process += 1

                # schedule more pages to keep window full (unless we know stop_at_page)
                while stop_at_page is None and len(in_flight) < max(1, workers) and within_limit(p):
                    # don't skip ahead too far; keep a tight rolling window
                    in_flight[p] = ex.submit(
                        fetch_page,
                        base_url, extra_params, p,
                        timeout, max_retries, backoff_base, backoff_max, jitter_s
                    )
                    p += 1

                # If we found stop_at_page, we can stop scheduling; just let in-flight drain
                if stop_at_page is not None:
                    for pg, f in list(in_flight.items()):
                        if pg > stop_at_page and f.cancel():
                            del in_flight[pg]

                    for pg in list(results_buffer.keys()):
                        if pg > stop_at_page:
                            del results_buffer[pg]
                break

            if not done_any:
                break
            
            if stop_at_page is not None and next_to_process > stop_at_page:
                break

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
    ap.add_argument("--params", default="", help='Extra query params like "mhc_class=I&disease=melanoma"')
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--max-pages", type=int, default=None)

    # speed / safety knobs
    ap.add_argument("--workers", type=int, default=4, help="Concurrent page fetchers (2-6 recommended)")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retries", type=int, default=6, help="Retries per page on 429/5xx/timeouts")
    ap.add_argument("--backoff-base", type=float, default=0.75, help="Base seconds for exponential backoff")
    ap.add_argument("--backoff-max", type=float, default=30.0, help="Max seconds to wait between retries")
    ap.add_argument("--jitter", type=float, default=0.25, help="Random jitter seconds added to waits")

    args = ap.parse_args()

    extra = parse_params_string(args.params)
    headers, rows = scrape(
        base_url=args.base_url,
        extra_params=extra,
        start_page=args.start_page,
        max_pages=args.max_pages,
        workers=args.workers,
        timeout=args.timeout,
        max_retries=args.retries,
        backoff_base=args.backoff_base,
        backoff_max=args.backoff_max,
        jitter_s=args.jitter,
    )

    write_csv(args.out, headers, rows)
    print(f"Saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
