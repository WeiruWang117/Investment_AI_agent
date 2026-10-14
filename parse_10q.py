"""
10-Q Parser: Extract MD&A (Item 2) and Risk Factors (Item 1A) from SEC quarterly filings.
Processes all tickers found in SEC_DIR.

10-Q structure differs from 10-K:
  Part I, Item 1  — Financial Statements
  Part I, Item 2  — Management's Discussion and Analysis (MD&A)
  Part I, Item 3  — Quantitative and Qualitative Disclosures About Market Risk
  Part I, Item 4  — Controls and Procedures
  Part II, Item 1 — Legal Proceedings
  Part II, Item 1A— Risk Factors (updates only)
  Part II, Item 2 — Unregistered Sales of Equity Securities
  ...
"""

from bs4 import BeautifulSoup
import re
import os
import warnings
warnings.filterwarnings("ignore")

SEC_DIR = "d:/Investment_AI_agent/SEC filings/sec-edgar-filings"
OUTPUT_DIR = "d:/Investment_AI_agent/parsed_10q"
MAX_CHARS = None  # No truncation — store full section, truncate later when passing to LLM


def html_to_text(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "lxml")
    return soup.get_text(separator=" ", strip=True)


def extract_section(text, start_pattern, end_pattern):
    """
    Find the best occurrence of start_pattern (skipping table of contents).
    Returns the text between start and the next end_pattern match.
    """
    starts = [m.start() for m in re.finditer(start_pattern, text, re.IGNORECASE)]

    if len(starts) == 0:
        return None

    # Skip first occurrence (table of contents); pick the occurrence with most content after it
    best_pos = None
    best_len = 0
    candidates = starts[1:] if len(starts) > 1 else starts  # fallback if only 1 hit

    for pos in candidates:
        ends = [m.start() for m in re.finditer(end_pattern, text[pos:], re.IGNORECASE)]
        section_len = ends[0] if ends else len(text) - pos
        if section_len > best_len:
            best_len = section_len
            best_pos = pos

    if best_pos is None:
        best_pos = starts[-1]

    remaining = text[best_pos:]
    end_matches = list(re.finditer(end_pattern, remaining, re.IGNORECASE))
    if end_matches:
        end_pos = best_pos + end_matches[0].start()
        section_text = text[best_pos:end_pos]
    else:
        section_text = text[best_pos:]

    return section_text.strip()


def get_period_label(folder_path):
    """
    Read CONFORMED PERIOD OF REPORT from full-submission.txt to get the exact
    reporting period end date, then derive the calendar quarter.

    Example: '20150926' → '2015Q3'
    Falls back to accession number year+seq if the file is missing.
    """
    submission_file = os.path.join(folder_path, "full-submission.txt")
    if os.path.exists(submission_file):
        with open(submission_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("CONFORMED PERIOD OF REPORT"):
                    date_str = line.split(":")[-1].strip()  # e.g. '20150926'
                    if len(date_str) == 8:
                        year  = date_str[:4]
                        month = int(date_str[4:6])
                        quarter = (month - 1) // 3 + 1
                        return f"{year}Q{quarter}"

    # Fallback: derive from accession number folder name
    folder_name = os.path.basename(folder_path)
    parts = folder_name.split("-")
    if len(parts) >= 2:
        yy = parts[1]
        year = "20" + yy if int(yy) <= 30 else "19" + yy
        seq = parts[2] if len(parts) >= 3 else "00"
        return f"{year}_{seq}"
    return folder_name


def parse_all_tickers():
    ticker_list = sorted(os.listdir(SEC_DIR))
    print(f"Found {len(ticker_list)} tickers: {ticker_list}\n")

    summary = {"success": 0, "warning": 0, "skipped": 0}

    # Negative lookahead — avoids matching cross-references like "discussed in Item 2 of this report"
    NO_XREF = r"(?!of\b|in\b|this\b|our\b|the\b)"

    for ticker in ticker_list:
        base_path = os.path.join(SEC_DIR, ticker, "10-Q")

        if not os.path.exists(base_path):
            print(f"[{ticker}] No 10-Q folder found, skipping.\n")
            summary["skipped"] += 1
            continue

        # Output dirs: parsed_10q/AMD/mda/  and  parsed_10q/AMD/risk/
        mda_dir  = os.path.join(OUTPUT_DIR, ticker, "mda")
        risk_dir = os.path.join(OUTPUT_DIR, ticker, "risk")
        os.makedirs(mda_dir,  exist_ok=True)
        os.makedirs(risk_dir, exist_ok=True)

        filing_folders = sorted(os.listdir(base_path))
        print(f"[{ticker}] {len(filing_folders)} 10-Q filings")

        for folder in filing_folders:
            html_path = os.path.join(base_path, folder, "primary-document.html")

            if not os.path.exists(html_path):
                print(f"  [{folder}] primary-document.html not found, skipping.")
                summary["skipped"] += 1
                continue

            label = get_period_label(os.path.join(base_path, folder))
            text = html_to_text(html_path)

            # ------------------------------------------------------------------
            # Extract MD&A — 10-Q uses Part I, Item 2
            # End boundary: Item 3 (Market Risk) or Item 4 (Controls)
            # ------------------------------------------------------------------
            # Primary: match by section title — avoids confusing Part I Item 2 (MD&A)
            # with Part II Item 2 (Unregistered Sales), which also starts with "Item 2"
            END_MDA = (
                r"[Qq]uantitative\s+and\s+[Qq]ualitative|"
                r"[Ii]tem\s*3[\s\.\:\u2014\-]" + NO_XREF +
                r"|[Ii]tem\s*4[\s\.\:\u2014\-]" + NO_XREF
            )
            mda = extract_section(
                text,
                start_pattern=r"[Mm]anagement.{0,5}s\s+[Dd]iscussion\s+and\s+[Aa]nalysis",
                end_pattern=END_MDA
            )
            # Fallback: use Item 2 header if title not found
            if not mda:
                mda = extract_section(
                    text,
                    start_pattern=r"[Ii]tem\s*2[\s\.\:\u2014\-]" + NO_XREF,
                    end_pattern=END_MDA
                )

            # ------------------------------------------------------------------
            # Extract Risk Factors — 10-Q uses Part II, Item 1A (updates only)
            # End boundary: Item 2 of Part II (Unregistered Sales…)
            # ------------------------------------------------------------------
            risk = extract_section(
                text,
                start_pattern=r"[Ii]tem\s*1[Aa][\s\.\:\u2014\-]" + NO_XREF,
                end_pattern=(
                    r"[Ii]tem\s*1[Bb][\s\.\:\u2014\-]" + NO_XREF +
                    r"|[Ii]tem\s*2[\s\.\:\u2014\-]" + NO_XREF +
                    r"|[Ii]tem\s*3[\s\.\:\u2014\-]" + NO_XREF
                )
            )

            # ------------------------------------------------------------------
            # Save sections
            # ------------------------------------------------------------------
            for section_name, content, out_dir in [
                ("mda",  mda,  mda_dir),
                ("risk", risk, risk_dir),
            ]:
                filename = f"{ticker}_{label}_{section_name}.txt"
                out_path = os.path.join(out_dir, filename)
                if content:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(content if MAX_CHARS is None else content[:MAX_CHARS])
                    summary["success"] += 1
                else:
                    print(f"  WARNING: [{ticker} {label}] could not find {section_name} section")
                    summary["warning"] += 1

        print()

    print("=" * 50)
    print(f"Done. Success: {summary['success']} | Warnings: {summary['warning']} | Skipped: {summary['skipped']}")


if __name__ == "__main__":
    parse_all_tickers()
