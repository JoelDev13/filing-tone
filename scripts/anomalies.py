import json, statistics
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORES = Path("data/scores"); MEAS = Path("measurements"); MEAS.mkdir(exist_ok=True)

def load_all() -> dict:
    out = {}
    for p in SCORES.glob("*.json"):
        ticker, year = p.stem.split("_")
        out.setdefault(ticker, {})[int(year)] = json.loads(p.read_text())
    return out

def detect(history: dict) -> list:
    tones = [history[y]["tone"] for y in sorted(history) if "tone" in history[y]]
    if len(tones) < 3:
        return []
    mean, sd = statistics.mean(tones), statistics.pstdev(tones) or 1e-9
    return [{"year": y,
             "z": round((history[y]["tone"] - mean) / sd, 2),
             "drift": history[y].get("drift_vs_prev")}
            for y in sorted(history)
            if "tone" in history[y]
            and (abs((history[y]["tone"] - mean) / sd) >= 1.5
                 or history[y].get("drift_vs_prev", 0) >= 0.5)]

def run():
    data = load_all()
    report = {t: detect(h) for t, h in data.items()}
    (MEAS / "anomalies.json").write_text(json.dumps(report, indent=2))
    for t, h in data.items():  
        ys = sorted(h)
        plt.plot(ys, [h[y].get("tone", 0) for y in ys], marker="o", label=t)
    plt.legend(fontsize=7); plt.ylabel("tone (pos - neg)")
    plt.savefig(MEAS / "tone_over_time.png", dpi=120)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    run()