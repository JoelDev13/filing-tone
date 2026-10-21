import re
import requests

UA = {"User-Agent": "filing-tone research filingtone.contact@proton.me"}

COMPANIES = {  # Ticker: CIK (10-digit number, zero-padded)
    "AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810",
    "TSLA": "0001318605", "AMZN": "0001018724", "GOOGL": "0001652044",
    "META": "0001326801", "JPM": "0000019617", "XOM": "0000034088",
    "WMT": "0000104169",
}

def chunk_text(text: str, limit: int = 1800) -> list:
    """Split the text into chunks that fit within the model's limit, without breaking sentences"""
    chunks, current = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if len(current) + len(s) > limit and current:
            chunks.append(current.strip())
            current = s
        else:
            current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks

def tone_from_labels(labels: list) -> dict:
    """tone = fraction of positive sentencesfraction of negative sentences. Simple and interpretable"""
    n = len(labels) or 1
    pos = labels.count("positive") / n
    neg = labels.count("negative") / n
    return {"pos": round(pos, 3), "neg": round(neg, 3),
            "neu": round(labels.count("neutral") / n, 3),
            "tone": round(pos - neg, 3), "n_chunks": len(labels)}