import json
import statistics
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MEAS = BASE_DIR / "measurements"
ANOMALIES_PATH = MEAS / "anomalies.json"
RESULTS_PATH = MEAS / "backtest_results.json"

def get_anomalies_path() -> Path:
    """Locate anomalies report with fallback to root measurements path"""
    if ANOMALIES_PATH.exists():
        return ANOMALIES_PATH
    alt_path = BASE_DIR.parent / "measurements" / "anomalies.json"
    if alt_path.exists():
        return alt_path
    return ANOMALIES_PATH

def get_results_path() -> Path:
    """Ensure output directory exists and return destination path"""
    MEAS.mkdir(parents=True, exist_ok=True)
    return RESULTS_PATH

def run_backtest(anomalies: dict, risk_free_rate: float = 4.0) -> dict:
    """
    Execute quantitative backtest on SEC 8K tone anomalies
    Strategy: Short selling equities when z score >= 1.50 (extreme negative tone anomaly)

    A short trade profits when the underlying equity price falls:
        Trade Return (%) = -Stock Return (%)
        Win condition: Stock Return < 0 (equity fell, short position won)
    """
    qualifying_trades = []

    for ticker, events in anomalies.items():
        if not isinstance(events, list):
            continue
        for ev in events:
            if not isinstance(ev, dict):
                continue
            z = ev.get("z")
            stock_ret = ev.get("stock_return_pct")

            # Filter for extreme tone anomaly: z >= 1.50 with valid numerical equity return
            if z is not None and z >= 1.50 and stock_ret is not None and isinstance(stock_ret, (int, float)):
                trade_return = -float(stock_ret)
                qualifying_trades.append({
                    "ticker": ticker,
                    "year": ev.get("year"),
                    "z": z,
                    "stock_return_pct": float(stock_ret),
                    "trade_return_pct": trade_return,
                    "is_win": stock_ret < 0
                })

    total_trades = len(qualifying_trades)

    if total_trades == 0:
        return {
            "strategy": "short_selling_z_score_ge_1.5",
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sharpe_ratio": 0.0,
            "risk_free_rate": risk_free_rate,
            "trades": []
        }

    wins = sum(1 for t in qualifying_trades if t["is_win"])
    win_rate = round((wins / total_trades) * 100.0, 2)

    returns = [t["trade_return_pct"] for t in qualifying_trades]
    avg_return = round(statistics.mean(returns), 2)

    if total_trades >= 2:
        std_dev = statistics.stdev(returns)
        sharpe_ratio = round((avg_return - risk_free_rate) / std_dev, 2) if std_dev > 1e-9 else 0.0
    else:
        std_dev = 0.0
        sharpe_ratio = 0.0

    return {
        "strategy": "short_selling_z_score_ge_1.5",
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "sharpe_ratio": sharpe_ratio,
        "risk_free_rate": risk_free_rate,
        "trades": qualifying_trades
    }

def print_summary_table(results: dict):
    """Print an institutional grade summary table of backtest results to console"""
    total = results.get("total_trades", 0)
    win_rate = results.get("win_rate", 0.0)
    avg_ret = results.get("avg_return", 0.0)
    sharpe = results.get("sharpe_ratio", 0.0)
    rf = results.get("risk_free_rate", 4.0)

    print("\n" + "=" * 62)
    print("           QUANTITATIVE ALPHA BACKTEST RESULTS")
    print("     Strategy: Short Selling on SEC 8K Anomaly (z >= 1.5)")
    print("=" * 62)
    print(f"  {'Metric':<35} {'Value':>22}")
    print("-" * 62)
    print(f"  {'Total Executed Trades':<35} {total:>22}")
    print(f"  {'Win Rate':<35} {win_rate:>21.2f}%")
    print(f"  {'Average Return (Short)':<35} {avg_ret:>+21.2f}%")
    print(f"  {'Benchmark Risk Free Rate':<35} {rf:>21.2f}%")
    print(f"  {'Annualized Sharpe Ratio':<35} {sharpe:>22.2f}")
    print("=" * 62 + "\n")

def run() -> dict:
    """Execute quantitative backtest workflow and serialize results"""
    anomalies_file = get_anomalies_path()
    if not anomalies_file.exists():
        print(f"[INFO] No anomaly report found at {anomalies_file}, skipping backtest")
        return {}

    try:
        content = anomalies_file.read_text(encoding="utf-8").strip()
        if not content:
            print(f"[INFO] Anomaly report at {anomalies_file} is empty, skipping backtest")
            return {}
        data = json.loads(content)
    except Exception as e:
        print(f"[WARNING] Failed to parse anomaly report: {e}")
        return {}

    results = run_backtest(data)
    print_summary_table(results)

    out_path = get_results_path()
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Backtest results serialized to: {out_path}")
    return results

if __name__ == "__main__":
    run()

