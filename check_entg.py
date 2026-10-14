from bs4 import BeautifulSoup

path = "d:/Investment_AI_agent/SEC filings/sec-edgar-filings/ENTG/10-Q/0001101302-15-000004/primary-document.html"
with open(path, encoding="utf-8", errors="ignore") as f:
    text = BeautifulSoup(f, "lxml").get_text(" ", strip=True)

idx = text.lower().find("item 1a")
if idx >= 0:
    print("Found Item 1A:")
    print(text[idx:idx+500])
else:
    print("Item 1A not found at all in document")
    # Show what items exist
    import re
    items = re.findall(r"[Ii]tem\s*\d+\w?[\s\.\:]", text)
    print("Items found:", list(dict.fromkeys(items))[:20])
