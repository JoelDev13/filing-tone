import json
from pathlib import Path
from transformers import pipeline
from common import chunk_text, tone_from_labels

BASE_DIR = Path(__file__).resolve().parent
RAW = BASE_DIR / "data" / "raw"
SCORES = BASE_DIR / "data" / "scores"; SCORES.mkdir(parents=True, exist_ok=True)
clf = pipeline("sentiment-analysis", model="ProsusAI/finbert")  # open source  

def score_file(path: Path) -> dict:
    chunks = chunk_text(path.read_text(encoding="utf-8"))
    results = clf(chunks, batch_size=8, truncation=True)
    labels = [r["label"] for r in results]
    scores = tone_from_labels(labels)

    # Extract the chunk with highest negative confidence as context snippet
    worst_score, worst_idx = 0.0, -1
    for i, r in enumerate(results):
        if r["label"] == "negative" and r["score"] > worst_score:
            worst_score, worst_idx = r["score"], i

    if worst_idx >= 0:
        snippet = chunks[worst_idx][:150]
        if len(chunks[worst_idx]) > 150:
            snippet += "..."
        scores["worst_snippet"] = snippet
    else:
        scores["worst_snippet"] = "No negative content detected"

    return scores

def run():
    for txt in sorted(RAW.glob("*.txt")):
        out = SCORES / f"{txt.stem}.json"
        if out.exists():
            continue
        s = score_file(txt)
        out.write_text(json.dumps(s, indent=2))
        print(txt.stem, s)

if __name__ == "__main__":
    run()