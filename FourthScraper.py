# Again for you fools
# Usage:
#   python FourthScraper.py --out psms.csv
#   python FourthScraper.py --out psms.csv --max-pages 1000
#   python FourthScraper.py --out psms.csv --params "mhc_class=I&disease=melanoma"

import argparse
import csv
import random
import threading
import time
import re
from urllib.parse import urljoin, parse_qsl
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

# ---------------- CONFIG ----------------

BASE_URL = "https://pci-db.org/psms/"
MAX_WORKERS = 10
MAX_RETRIES = 5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

FINAL_HEADERS = [
    "ID", "Peptide Sequence", "Best HLA Allele", "Tissue", "Disease",
    "MHC Class", "Peptide Modifications", "Uniprot IDs", "Affinity % Rank",
    "Charge", "XCorr", "Q Value", "Matched Ions", "Total Ions", "Image File"
]

_thread_local = threading.local()

# FAILED_IDS[row_id] = {url, attempts}
FAILED_IDS = {}

# ---------------- SESSION ----------------

def _get_session():
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(HEADERS)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=50,
            pool_maxsize=50
        )
        sess.mount("https://", adapter)
        _thread_local.session = sess
    return sess

# ---------------- UTILS ----------------

def clean_key(s):
    return re.sub(r"[^a-zA-Z0-9]", "", (s or "").lower())

def clean_text(s):
    return " ".join((s or "").split()).strip()

# ---------------- DETAIL FETCH ----------------

def fetch_spectrum_details(detail_url, row_id):
    session = _get_session()
    time.sleep(random.uniform(0.2, 0.6))

    details = {
        "Charge": "",
        "XCorr": "",
        "Q Value": "",
        "Matched Ions": "",
        "Total Ions": "",
        "Image File": "",
    }

    try:
        r = session.get(detail_url, timeout=30)
        if r.status_code != 200:
            raise Exception("Bad status")

        soup = BeautifulSoup(r.text, "html.parser")

        for tbl in soup.find_all("table"):
            for tr in tbl.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                if len(cells) >= 2:
                    k = clean_key(cells[0].get_text())
                    v = clean_text(cells[1].get_text())

                    if k == "charge":
                        details["Charge"] = v
                    elif k == "xcorr":
                        details["XCorr"] = v
                    elif k == "qvalue":
                        details["Q Value"] = v
                    elif k == "matchedions":
                        details["Matched Ions"] = v
                    elif k == "totalions":
                        details["Total Ions"] = v

        if not any(details.values()):
            raise Exception("Empty")

        return details

    except:
        entry = FAILED_IDS.setdefault(row_id, {"url": detail_url, "attempts": 0})
        entry["attempts"] += 1
        return None

# ---------------- RETRY PER PAGE ----------------

def retry_failed_for_rows(rows):
    """
    Retry failed peptides that belong to THIS PAGE only.
    """
    for row in rows:
        rid = int(row["ID"])
        if rid not in FAILED_IDS:
            continue

        entry = FAILED_IDS[rid]
        if entry["attempts"] >= MAX_RETRIES:
            continue

        result = fetch_spectrum_details(entry["url"], rid)
        if result:
            row.update(result)
            FAILED_IDS.pop(rid, None)

# ---------------- PAGE FETCH ----------------

def fetch_page_data(base_url, params, page_num, start_id):
    session = _get_session()

    try:
        p = dict(params)
        p["page"] = str(page_num)
        r = session.get(base_url, params=p, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")
    except:
        return [], False

    main_table = None
    for t in soup.find_all("table"):
        if "Peptide Sequence" in [
            clean_text(th.get_text()) for th in t.find_all("th")
        ]:
            main_table = t
            break

    if not main_table:
        return [], False

    headers = [clean_text(th.get_text()) for th in main_table.find_all("th")]
    tr_list = main_table.find_all("tr")[1:]

    print(f"Page {page_num}: Found {len(tr_list)} peptides")

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = []
    rows = []
    cid = start_id

    for tr in tr_list:
        tds = tr.find_all("td")
        if not tds:
            continue

        vals = [clean_text(td.get_text()) for td in tds]
        row = dict(zip(headers, vals))
        row_id = cid
        cid += 1

        link = tr.find("a", string=re.compile("Spectrum Info", re.I)) or \
               tr.find("a", class_="btn-info")

        if link and link.get("href"):
            detail_url = urljoin(base_url, link.get("href"))
            future = executor.submit(fetch_spectrum_details, detail_url, row_id)
            futures.append((future, row, row_id))
        else:
            futures.append((None, row, row_id))

    for future, row, row_id in futures:
        if future:
            result = future.result()
            if result:
                row.update(result)

        clean_row = {c: row.get(c, "") for c in FINAL_HEADERS}
        clean_row["ID"] = str(row_id)
        rows.append(clean_row)

    executor.shutdown(wait=True)

    has_next = any(
        "next" in clean_text(a.get_text()).lower()
        for a in soup.find_all("a")
    )

    return rows, has_next

# ---------------- MAIN ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--params", default="")
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()

    params = dict(parse_qsl(args.params)) if args.params else {}

    all_rows = []
    page = 1
    gid = 1

    while True:
        if args.max_pages and page > args.max_pages:
            break

        print(f"--- Starting Page {page} ---")

        rows, has_next = fetch_page_data(BASE_URL, params, page, gid)
        if not rows:
            break

        all_rows.extend(rows)
        gid += len(rows)

        # 🔁 Retry failures from THIS PAGE immediately
        for _ in range(MAX_RETRIES):
            retry_failed_for_rows(rows)

        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FINAL_HEADERS)
            w.writeheader()
            w.writerows(all_rows)

        if not has_next:
            break

        page += 1
        time.sleep(1)

    print(f"Done. Saved {len(all_rows)} rows.")
    if FAILED_IDS:
        print(f"Permanent failures: {len(FAILED_IDS)}")

if __name__ == "__main__":
    main()
