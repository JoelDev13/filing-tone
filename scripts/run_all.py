import fetch_text, analyze_tone, drift, anomalies, notify

for mod in (fetch_text, analyze_tone, drift, anomalies, notify):
    mod.run()