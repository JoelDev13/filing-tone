import json
import logging
import warnings
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress third party deprecation warnings (e.x. pandas/yfinance internal warnings)
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

try:
    import yfinance as yf
except ImportError:
    yf = None

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

def get_stock_return(ticker: str, year: int) -> float:
    """
    Calculate the simple annual percentage return for an equity ticker:
        Return % = ((Close_Final - Close_Initial) / Close_Initial) * 100

    Financial Ground Truth:
    Validates whether statistical tone anomalies detected in SEC 8K filings
    coincide with actual equity price movements over the reporting period.
    Returns 0.0 defensively if market data is unavailable, invalid, or upon API/network error
    """
    try:
        if yf is None:
            return 0.0

        # Download historical equity price data for the specified calendar year
        # (end date is non inclusive in yfinance, so year+1 covers through Dec 31)
        start_date = f"{year}-01-01"
        end_date = f"{year + 1}-01-01"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                timeout=15,
            )

        if data is None or data.empty:
            return 0.0

        # Extract Close price series safely, accommodating single index and MultiIndex formats
        if "Close" in data.columns:
            close_col = data["Close"]
        elif "Adj Close" in data.columns:
            close_col = data["Adj Close"]
        else:
            close_col = data.iloc[:, 0]

        # Handle 2D DataFrame / MultiIndex produced by newer yfinance releases
        if getattr(close_col, "ndim", 1) == 2:
            if hasattr(close_col, "columns") and ticker in close_col.columns:
                close_series = close_col[ticker].dropna()
            else:
                close_series = close_col.iloc[:, 0].dropna()
        else:
            close_series = close_col.dropna()

        if len(close_series) < 2:
            return 0.0

        p_initial = float(close_series.iloc[0])
        p_final = float(close_series.iloc[-1])

        if p_initial <= 0.0:
            return 0.0

        return_pct = ((p_final - p_initial) / p_initial) * 100.0
        return round(float(return_pct), 2)

    except Exception as e:
        # Non blocking defensive error handling: never crash the NLP pipeline
        print(f"  [WARNING] Unable to fetch market return for {ticker} ({year}): {e}")
        return 0.0

def detect(history: dict, ticker: str = "") -> list:
    """
    Detect statistical tone anomalies using centralized decision boundaries
    Correlates detected anomalies with equity market ground truth (annual stock return %).
    Maintains 100% backward compatibility with pipeline notifications
    """
    stats = compute_ticker_statistics(history)
    anomalies = []
    for r in stats["records"]:
        if r["is_anomaly"]:
            year = r["year"]
            # Fetch ground truth stock return if not already cached in historical records
            stock_ret = history.get(year, {}).get("stock_return_pct")
            if stock_ret is None and ticker:
                stock_ret = get_stock_return(ticker, year)
                if year in history:
                    history[year]["stock_return_pct"] = stock_ret

            anomalies.append({
                "year": year,
                "z": r["z"],
                "drift": r["drift"],
                "worst_snippet": r["full_snippet"] if r["full_snippet"] != EMPTY_SNIPPET_FALLBACK else "",
                "stock_return_pct": stock_ret if stock_ret is not None else 0.0
            })
    return anomalies

def run():
    """Execute anomaly detection pipeline and regenerate dashboard"""
    data = load_all()
    report = {t: detect(h, ticker=t) for t, h in data.items()}
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
