import json
from pathlib import Path
from transformers import pipeline
from common import chunk_text, tone_from_labels

RAW = Path("data/raw")
SCORES = Path("data/scores"); SCORES.mkdir(parents=True, exist_ok=True)
clf = pipeline("sentiment-analysis", model="ProsusAI/finbert")  # open source  

def score_file(path: Path) -> dict:
    chunks = chunk_text(path.read_text(encoding="utf-8"))
    labels = [r["label"] for r in clf(chunks, batch_size=8, truncation=True)]
    return tone_from_labels(labels)

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