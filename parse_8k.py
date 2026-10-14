"""
8-K Parser: Extract text from SEC 8-K filings.

Reads from:  SEC filings/sec-edgar-filings/{TICKER}/8-K/{accession}/
Writes to:   parsed_8k/{TICKER}/{YYYYMMDD}_{accession}.txt

Each output file starts with a metadata header (filing date, items),
followed by the full text of the primary document.

Only investment-relevant Item types are kept:
  1.01  Entry into Material Agreement
  2.02  Results of Operations (earnings)
  5.02  Management Changes
  7.01  Regulation FD / guidance
  8.01  Other Material Events
"""

import os
import re
import warnings
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

SEC_DIR    = "d:/Investment_AI_agent/SEC filings/sec-edgar-filings"
OUTPUT_DIR = "d:/Investment_AI_agent/parsed_8k"

# Only keep 8-Ks that contain at least one of these items
RELEVANT_ITEMS = {"1.01", "2.02", "5.02", "7.01", "8.01"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_header(submission_path: str) -> dict:
    """
    Extract filing date and item list from full-submission.txt header.
    Returns {"date": "YYYYMMDD", "items": ["2.02", ...]} or None on failure.
    """
    try:
        with open(submission_path, "r", encoding="utf-8", errors="ignore") as f:
            # Header ends at first <DOCUMENT> tag — read only that portion
            header = ""
            for line in f:
                if line.strip().startswith("<DOCUMENT>"):
                    break
                header += line
    except OSError:
        return None

    # Filing date
    date_match = re.search(r"FILED AS OF DATE:\s*(\d{8})", header)
    if not date_match:
        return None
    filed_date = date_match.group(1)

    # Item types  e.g. "ITEM INFORMATION:  Results of Operations ..."
    # Map verbose descriptions to short item numbers
    item_lines = re.findall(r"ITEM INFORMATION:\s*(.+)", header)
    items = []
    for line in item_lines:
        line = line.strip()
        # Try to extract item number from the text (e.g. "Item 2.02" or just look up)
        num_match = re.search(r"\b(\d+\.\d+)\b", line)
        if num_match:
            items.append(num_match.group(1))
        else:
            # Fallback: keyword mapping for common verbose descriptions
            lower = line.lower()
            if "results of operations" in lower:
                items.append("2.02")
            elif "regulation fd" in lower:
                items.append("7.01")
            elif "other events" in lower:
                items.append("8.01")
            elif "material definitive agreement" in lower or "entry into" in lower:
                items.append("1.01")
            elif "departure" in lower or "appointment" in lower or "officer" in lower:
                items.append("5.02")

    return {"date": filed_date, "items": items}


def html_to_text(html_path: str) -> str:
    """Strip HTML tags and return clean plain text."""
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(separator="\n", strip=True)


# ── Main Parser ────────────────────────────────────────────────────────────────

def parse_all_tickers():
    ticker_list = sorted(os.listdir(SEC_DIR))
    print(f"Found {len(ticker_list)} tickers: {ticker_list}\n")

    total_saved = total_skipped = total_irrelevant = 0

    for ticker in ticker_list:
        eightk_dir = os.path.join(SEC_DIR, ticker, "8-K")
        if not os.path.exists(eightk_dir):
            print(f"[{ticker}] No 8-K folder, skipping.")
            continue

        out_dir = os.path.join(OUTPUT_DIR, ticker)
        os.makedirs(out_dir, exist_ok=True)

        filings = sorted(os.listdir(eightk_dir))
        saved = skipped = irrelevant = 0

        for accession in filings:
            folder       = os.path.join(eightk_dir, accession)
            sub_path     = os.path.join(folder, "full-submission.txt")
            html_path    = os.path.join(folder, "primary-document.html")

            if not os.path.exists(sub_path) or not os.path.exists(html_path):
                skipped += 1
                continue

            meta = parse_header(sub_path)
            if not meta:
                skipped += 1
                continue

            # Filter: skip if no investment-relevant items
            relevant = [it for it in meta["items"] if it in RELEVANT_ITEMS]
            if not relevant:
                irrelevant += 1
                continue

            out_filename = f"{ticker}_{meta['date']}.txt"
            out_path     = os.path.join(out_dir, out_filename)

            if os.path.exists(out_path):
                saved += 1
                continue  # already parsed

            text = html_to_text(html_path)
            if not text.strip():
                skipped += 1
                continue

            items_str = ", ".join(relevant)
            header_block = (
                f"TICKER:      {ticker}\n"
                f"FILED:       {meta['date']}\n"
                f"ACCESSION:   {accession}\n"
                f"ITEMS:       {items_str}\n"
                f"{'─' * 60}\n\n"
            )

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(header_block + text)

            saved += 1

        print(f"[{ticker}]  saved={saved}  irrelevant={irrelevant}  skipped={skipped}")
        total_saved     += saved
        total_skipped   += skipped
        total_irrelevant += irrelevant

    print(f"\n{'=' * 50}")
    print(f"Done.  Saved: {total_saved} | Irrelevant: {total_irrelevant} | Skipped: {total_skipped}")


if __name__ == "__main__":
    parse_all_tickers()
