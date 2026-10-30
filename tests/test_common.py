from unittest.mock import patch, MagicMock
from scripts.common import chunk_text, tone_from_labels, is_anomaly
from scripts.anomalies import get_stock_return, detect
from scripts.notify import _send_telegram

def test_tone_from_labels():
    r = tone_from_labels(["positive", "negative", "neutral", "positive"])
    assert r["pos"] == 0.5 and r["neg"] == 0.25 and r["tone"] == 0.25

def test_tone_empty_no_crash():
    assert tone_from_labels([])["tone"] == 0

def test_chunks_respect_limit():
    text = ". ".join(f"sentence number {i}" for i in range(200))
    chunks = chunk_text(text, limit=500)
    assert len(chunks) > 1
    assert sum(c.count("sentence") for c in chunks) == 200

def test_is_anomaly_decision_boundary():
    # |z| >= 1.5 triggers anomaly with n >= 3
    assert is_anomaly(z_score=1.71, drift=0.36, n_samples=4) is True
    assert is_anomaly(z_score=-1.60, drift=0.01, n_samples=3) is True
    # drift >= 0.5 triggers anomaly
    assert is_anomaly(z_score=0.10, drift=0.55, n_samples=3) is True
    # Nominal values do not trigger
    assert is_anomaly(z_score=0.14, drift=0.002, n_samples=3) is False
    assert is_anomaly(z_score=-1.40, drift=0.003, n_samples=3) is False

def test_is_anomaly_insufficient_history():
    # Requires at least 3 historical observations
    assert is_anomaly(z_score=2.50, drift=0.80, n_samples=2) is False
    assert is_anomaly(z_score=2.50, drift=0.80, n_samples=1) is False

def test_get_stock_return_handles_api_exception():
    with patch("scripts.anomalies.yf") as mock_yf:
        mock_yf.download.side_effect = ConnectionError("Network failure")
        ret = get_stock_return("AAPL", 2024)
        assert ret == 0.0

def test_get_stock_return_handles_none_yf():
    with patch("scripts.anomalies.yf", None):
        ret = get_stock_return("AAPL", 2024)
        assert ret == 0.0

def test_get_stock_return_math_calculation():
    mock_series = MagicMock()
    mock_series.ndim = 1
    mock_series.dropna.return_value = mock_series
    mock_series.__len__.return_value = 5
    # Initial price: 100.0, Final price: 125.0 -> +25.0%
    mock_series.iloc = MagicMock()
    mock_series.iloc.__getitem__.side_effect = lambda idx: 100.0 if idx == 0 else 125.0

    mock_df = MagicMock()
    mock_df.empty = False
    mock_df.columns = ["Close"]
    mock_df.__getitem__.return_value = mock_series

    with patch("scripts.anomalies.yf") as mock_yf:
        mock_yf.download.return_value = mock_df
        ret = get_stock_return("AAPL", 2024)
        assert ret == 25.0

def test_detect_enriches_stock_return():
    history = {
        2023: {"tone": 0.10, "worst_snippet": "All good"},
        2024: {"tone": 0.12, "worst_snippet": "Stable"},
        2025: {"tone": 0.11, "worst_snippet": "Normal"},
        2026: {"tone": -0.80, "worst_snippet": "Critical negative event", "drift_vs_prev": 0.60}
    }
    with patch("scripts.anomalies.get_stock_return", return_value=-15.5):
        anomalies = detect(history, ticker="AAPL")
        assert len(anomalies) == 1
        anom = anomalies[0]
        assert anom["year"] == 2026
        assert "stock_return_pct" in anom
        assert anom["stock_return_pct"] == -15.5

def test_notify_message_format():
    with patch("scripts.notify.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        anomaly = {
            "year": 2026,
            "z": 1.71,
            "drift": 0.36,
            "worst_snippet": "Negative disclosure excerpt",
            "stock_return_pct": 14.5
        }
        _send_telegram("token123", "chat123", "AAPL", anomaly)
        call_json = mock_post.call_args[1]["json"]
        assert "*Annual Equity Return:* +14.50%" in call_json["text"]
        assert "*Ticker:* `AAPL` (2026)" in call_json["text"]
        assert "*Z Score:*" in call_json["text"]

def test_notify_message_format_missing_key():
    with patch("scripts.notify.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        anomaly = {
            "year": 2025,
            "z": 1.55,
            "drift": 0.52,
            "worst_snippet": "Negative disclosure excerpt"
            # No "stock_return_pct" key
        }
        _send_telegram("token123", "chat123", "MSFT", anomaly)
        call_json = mock_post.call_args[1]["json"]
        assert "*Annual Equity Return:* 0%" in call_json["text"]
