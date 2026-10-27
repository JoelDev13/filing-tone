import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from common import chunk_text, COMPANIES

BASE_DIR = Path(__file__).resolve().parent
RAW = BASE_DIR / "data" / "raw"; SCORES = BASE_DIR / "data" / "scores"
model = SentenceTransformer("all-MiniLM-L6-v2")  

def doc_vector(path: Path) -> np.ndarray:
    chunks = chunk_text(path.read_text(encoding="utf-8"), limit=900)  # MiniLM: 256 tokens
    return model.encode(chunks, batch_size=16).mean(axis=0)

def run():
    for ticker in COMPANIES:
        files = sorted(RAW.glob(f"{ticker}_*.txt"))
        vecs = {f.stem.split("_")[1]: doc_vector(f) for f in files}
        years = sorted(vecs)
        for a, b in zip(years, years[1:]):
            sim = float(np.dot(vecs[a], vecs[b]) / (np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b])))
            p = SCORES / f"{ticker}_{b}.json"
            s = json.loads(p.read_text()) if p.exists() else {}
            s["drift_vs_prev"] = round(1 - sim, 3)
            s["drift_baseline"] = a
            p.write_text(json.dumps(s, indent=2))
            print(ticker, b, "drift", s["drift_vs_prev"])

if __name__ == "__main__":
    run()