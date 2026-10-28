from scripts.common import chunk_text, tone_from_labels, is_anomaly

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