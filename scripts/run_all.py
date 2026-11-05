try:
    import fetch_text, analyze_tone, drift, anomalies, notify, backtest
except ImportError:
    from scripts import fetch_text, analyze_tone, drift, anomalies, notify, backtest

for mod in (fetch_text, analyze_tone, drift, anomalies, notify, backtest):
    mod.run()
