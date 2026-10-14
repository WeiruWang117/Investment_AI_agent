"""
10-K Parser: Extract MD&A (Item 7) and Risk Factors (Item 1A) from SEC filings.
Processes all tickers found in SEC_DIR.

"""

from bs4 import BeautifulSoup
import re
import os
import warnings
warnings.filterwarnings("ignore")

SEC_DIR = "d:/Investment_AI_agent/SEC filings/sec-edgar-filings"
OUTPUT_DIR = "d:/Investment_AI_agent/parsed_10k"
MAX_CHARS = None  # No truncation — store full section, truncate later when passing to LLM


def html_to_text(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "lxml")
    return soup.get_text(separator=" ", strip=True)


def extract_section(text, start_pattern, end_pattern):
    """
    Find the LAST occurrence of start_pattern before end_pattern.
    Skip the first occurrence (table of contents).
    """
    starts = [m.start() for m in re.finditer(start_pattern, text, re.IGNORECASE)]

    if len(starts) == 0:
        return None

    # Skip first occurrence (table of contents), use the one with most content after it
    best_pos = None
    best_len = 0
    for pos in starts[1:]:
        ends = [m.start() for m in re.finditer(end_pattern, text[pos:], re.IGNORECASE)]
        if ends:
            section_len = ends[0]
        else:
            section_len = len(text) - pos
        if section_len > best_len:
            best_len = section_len
            best_pos = pos

    if best_pos is None:
        best_pos = starts[-1]

    # Extract text from section start to next major Item
    remaining = text[best_pos:]
    end_matches = list(re.finditer(end_pattern, remaining, re.IGNORECASE))
    if end_matches:
        end_pos = best_pos + end_matches[0].start()
        section_text = text[best_pos:end_pos]
    else:
        section_text = text[best_pos:]

    return section_text.strip()


def folder_to_year(folder_name):
    """
    Extract 4-digit filing year from folder name.
    Example: '0001045810-23-000017' → '2023'
    """
    parts = folder_name.split("-")
    if len(parts) >= 2:
        yy = parts[1]
        year = "20" + yy if int(yy) <= 30 else "19" + yy
        return year
    return folder_name


def parse_all_tickers():
    # Auto-detect all tickers from SEC_DIR
    ticker_list = sorted(os.listdir(SEC_DIR))
    print(f"Found {len(ticker_list)} tickers: {ticker_list}\n")

    summary = {"success": 0, "warning": 0, "skipped": 0}

    for ticker in ticker_list:
        base_path = os.path.join(SEC_DIR, ticker, "10-K")

        if not os.path.exists(base_path):
            print(f"[{ticker}] No 10-K folder found, skipping.\n")
            summary["skipped"] += 1
            continue

        # Create per-ticker subfolders: parsed_10k/NVDA/mda/ and parsed_10k/NVDA/risk/
        mda_dir  = os.path.join(OUTPUT_DIR, ticker, "mda")
        risk_dir = os.path.join(OUTPUT_DIR, ticker, "risk")
        os.makedirs(mda_dir,  exist_ok=True)
        os.makedirs(risk_dir, exist_ok=True)

        filing_folders = sorted(os.listdir(base_path))
        print(f"[{ticker}] {len(filing_folders)} filings")

        for folder in filing_folders:
            html_path = os.path.join(base_path, folder, "primary-document.html")

            if not os.path.exists(html_path):
                print(f"  [{folder}] primary-document.html not found, skipping.")
                summary["skipped"] += 1
                continue

            year = folder_to_year(folder)
            text = html_to_text(html_path)

            # Negative lookahead (?!of\b|in\b|this|our) excludes cross-references like
            # "contained in Item 8 of this Annual Report"
            # and only matches standalone section headers like "Item 8. Financial Statements"
            NO_XREF = r"(?!of\b|in\b|this\b|our\b|the\b)"

            # Extract Risk Factors (Item 1A)
            risk = extract_section(
                text,
                start_pattern=r"[Ii]tem\s*1[Aa][\s\.\:\u2014\-]" + NO_XREF,
                end_pattern=r"[Ii]tem\s*1[Bb][\s\.\:\u2014\-]" + NO_XREF + r"|[Ii]tem\s*2[\s\.\:\u2014\-]" + NO_XREF
            )

            # Extract MD&A (Item 7), with fallback for non-standard formatting
            mda = extract_section(
                text,
                start_pattern=r"[Ii]tem\s*7[\s\.\:\u2014\-]" + NO_XREF,
                end_pattern=r"[Ii]tem\s*7[Aa][\s\.\:\u2014\-]" + NO_XREF + r"|[Ii]tem\s*8[\s\.\:\u2014\-]" + NO_XREF
            )
            if not mda:
                mda = extract_section(
                    text,
                    start_pattern=r"[Mm]anagement.{0,5}s [Dd]iscussion and [Aa]nalysis",
                    end_pattern=r"[Qq]uantitative and [Qq]ualitative|[Ii]tem\s*8[\s\.\:\u2014\-]" + NO_XREF
                )

            # Save to parsed_10k/NVDA/mda/ and parsed_10k/NVDA/risk/
            for section_name, content, out_dir in [
                ("mda",  mda,  mda_dir),
                ("risk", risk, risk_dir),
            ]:
                filename = f"{ticker}_{year}_{section_name}.txt"
                out_path = os.path.join(out_dir, filename)
                if content:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(content if MAX_CHARS is None else content[:MAX_CHARS])
                    summary["success"] += 1
                else:
                    print(f"  WARNING: [{ticker} {year}] could not find {section_name} section")
                    summary["warning"] += 1

        print()

    print("=" * 50)
    print(f"Done. Success: {summary['success']} | Warnings: {summary['warning']} | Skipped: {summary['skipped']}")


if __name__ == "__main__":
    parse_all_tickers()
