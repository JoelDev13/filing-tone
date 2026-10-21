import requests
from bs4 import BeautifulSoup
from pathlib import Path
from common import UA, COMPANIES

RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)

def recent_8ks(cik: str, n: int = 2) -> list:
    d = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                     headers=UA, timeout=60).json()
    r = d["filings"]["recent"]
    out = []
    for form, date, acc, doc in zip(r["form"], r["filingDate"],
                                    r["accessionNumber"], r["primaryDocument"]):
        if form == "8-K": 
            out.append({"date": date, "accession": acc, "doc": doc})
            if len(out) == n:
                break
    return out

def filing_url(cik: str, f: dict) -> str:
    return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
            f"/{f['accession'].replace('-', '')}/{f['doc']}")

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "table"]):   # Financial tables = noise for sentiment analysis
        tag.decompose()
    return " ".join(soup.get_text(" ").split())

def run():
    for ticker, cik in COMPANIES.items():
        for f in recent_8ks(cik):
            dest = RAW / f"{ticker}_{f['date'][:4]}.txt"
            if dest.exists():
                continue
            r = requests.get(filing_url(cik, f), headers=UA, timeout=120)
            r.raise_for_status()
            dest.write_text(html_to_text(r.text), encoding="utf-8")
            print("saved", dest.name, len(dest.read_text(encoding="utf-8")), "chars")

if __name__ == "__main__":
    run()