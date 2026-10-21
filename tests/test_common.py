from scripts.common import chunk_text, tone_from_labels

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