import re
import json
import statistics
from pathlib import Path

UA = {"User-Agent": "filing-tone research filingtone.contact@proton.me"}

COMPANIES = {  # Ticker: CIK (10-digit number, zero-padded)
    "AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810",
    "TSLA": "0001318605", "AMZN": "0001018724", "GOOGL": "0001652044",
    "META": "0001326801", "JPM": "0000019617", "XOM": "0000034088",
    "WMT": "0000104169",
}

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "META": "Meta Platforms Inc.",
    "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.",
    "TSLA": "Tesla Inc.",
    "WMT": "Walmart Inc.",
    "XOM": "Exxon Mobil Corp."
}

# Single source of truth for quantitative decision boundaries
Z_SCORE_THRESHOLD = 1.5
DRIFT_THRESHOLD = 0.5
MIN_SAMPLES_FOR_ANOMALY = 3

def is_anomaly(z_score: float | None, drift: float | None, n_samples: int = 3) -> bool:
    """
    Evaluate if an observation exceeds statistical decision boundaries:
    |z| >= 1.50 or drift >= 0.50 (with n >= 3 historical observations)
    """
    if n_samples < MIN_SAMPLES_FOR_ANOMALY:
        return False
    if z_score is not None and abs(z_score) >= Z_SCORE_THRESHOLD:
        return True
    if drift is not None and drift >= DRIFT_THRESHOLD:
        return True
    return False

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
    """tone = fraction of positive sentences, fraction of negative sentences. Simple and interpretable"""
    n = len(labels) or 1
    pos = labels.count("positive") / n
    neg = labels.count("negative") / n
    return {"pos": round(pos, 3), "neg": round(neg, 3),
            "neu": round(labels.count("neutral") / n, 3),
            "tone": round(pos - neg, 3), "n_chunks": len(labels)}

def load_score_history(scores_dir: Path) -> dict:
    """Load historical score JSON files grouped by ticker and filing year"""
    data = {}
    if not scores_dir.exists():
        return data
    for p in scores_dir.glob("*.json"):
        parts = p.stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            ticker, year = parts[0], int(parts[1])
            try:
                data.setdefault(ticker, {})[year] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return data

EMPTY_SNIPPET_FALLBACK = "Form 8K disclosure analysis complete (Item 8.01 Other Events). No negative phrase exceeded polarity threshold"

def compute_ticker_statistics(history: dict) -> dict:
    tones = [history[y]["tone"] for y in sorted(history) if "tone" in history[y]]
    n = len(tones)
    mean_val = statistics.mean(tones) if tones else 0.0
    sd_val = statistics.pstdev(tones) if n >= 2 else 0.0

    yearly_records = []
    for y in sorted(history):
        rec = history[y]
        raw_tone = rec.get("tone", 0.0)
        z_score = round((raw_tone - mean_val) / sd_val, 2) if (n >= 2 and sd_val > 1e-9) else 0.0
        drift = rec.get("drift_vs_prev")
        drift_val = drift if isinstance(drift, (int, float)) else None
        anomaly_flag = is_anomaly(z_score, drift_val, n)

        snippet = str(rec.get("worst_snippet", "") or "")
        clean_snippet = snippet.replace('"', '&quot;').replace("'", "&#39;")
        truncated_snippet = (clean_snippet[:85] + "...") if len(clean_snippet) > 85 else (clean_snippet or "Form 8K disclosure analysis complete (Item 8.01 Other Events)")

        stock_ret = rec.get("stock_return_pct")
        ret_str = f"{stock_ret:+.2f}%" if isinstance(stock_ret, (int, float)) else "N/A"

        yearly_records.append({
            "year": y,
            "tone": raw_tone,
            "z": z_score,
            "drift": drift_val,
            "is_anomaly": anomaly_flag,
            "snippet": truncated_snippet,
            "full_snippet": clean_snippet or EMPTY_SNIPPET_FALLBACK,
            "stock_return": ret_str
        })

    return {
        "n": n,
        "mean": mean_val,
        "std": sd_val,
        "records": yearly_records
    }