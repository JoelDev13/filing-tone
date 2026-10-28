import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scripts.common import load_score_history, compute_ticker_statistics, EMPTY_SNIPPET_FALLBACK
    from scripts.dashboard import generate_dashboard
except ImportError:
    from common import load_score_history, compute_ticker_statistics, EMPTY_SNIPPET_FALLBACK
    from dashboard import generate_dashboard

BASE_DIR = Path(__file__).resolve().parent
SCORES = BASE_DIR / "data" / "scores"
MEAS = BASE_DIR / "measurements"; MEAS.mkdir(exist_ok=True)

def load_all() -> dict:
    """Load all score JSON files grouped by ticker and filing year"""
    return load_score_history(SCORES)

def detect(history: dict) -> list:
    """
    Detect statistical tone anomalies using centralized decision boundaries
    Maintains 100% backward compatibility with pipeline notifications
    """
    stats = compute_ticker_statistics(history)
    return [
        {
            "year": r["year"],
            "z": r["z"],
            "drift": r["drift"],
            "worst_snippet": r["full_snippet"] if r["full_snippet"] != EMPTY_SNIPPET_FALLBACK else "",
            "stock_return_pct": history[r["year"]].get("stock_return_pct")
        }
        for r in stats["records"]
        if r["is_anomaly"]
    ]

def run():
    """Execute anomaly detection pipeline and regenerate dashboard"""
    data = load_all()
    report = {t: detect(h) for t, h in data.items()}
    (MEAS / "anomalies.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Static matplotlib generation for backwards compatibility
    for t, h in data.items():
        ys = sorted(h)
        plt.plot(ys, [h[y].get("tone", 0) for y in ys], marker="o", label=t)
    plt.legend(fontsize=7); plt.ylabel("tone (pos - neg)")
    plt.savefig(MEAS / "tone_over_time.png", dpi=120)
    plt.close()

    print(json.dumps(report, indent=2))
    generate_dashboard(report=report, history_data=data)

if __name__ == "__main__":
    run()