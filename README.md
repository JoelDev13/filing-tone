# Filing Tone: Automated Financial Sentiment, Narrative Drift and Market Ground Truth in SEC 8K Disclosures

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/NLP-FinBERT-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/ProsusAI/finbert)
[![Embeddings](https://img.shields.io/badge/Embeddings-Sentence_Transformers-blue?style=flat-square)](https://sbert.net/)
[![Market Data](https://img.shields.io/badge/Market%20Data-yfinance-410099?style=flat-square&logo=yahoo&logoColor=white)](https://pypi.org/project/yfinance/)
[![Alerts](https://img.shields.io/badge/Alerts-Telegram-2CA5E0?style=flat-square&logo=telegram&logoColor=white)](https://core.telegram.org/bots/api)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## Abstract

**Filing Tone** is an autonomous serverless computational framework for longitudinal financial text mining and equity market ground truth validation. The system continuously ingests unscheduled material event disclosures (Form 8K) from the U.S. Securities and Exchange Commission (SEC) EDGAR system, extracts sentence level financial sentiment via domain adapted transformer models (**FinBERT**), and tracks semantic narrative drift over time using dense vector representations (**SBERT**). 

Statistical outliers are identified via standardized z score thresholds and cosine drift distances, enriched with quantitative equity market ground truth via **Yahoo Finance (`yfinance`)**, augmented with maximal negative contextual excerpts, and dispatched via idempotent real time event notifications. All historical data, model artifacts, and interactive Bloomberg and FactSet style terminal visualizers are persisted under a Git as a Datastore paradigm.

---

## Methodology and Mathematical Formulation

The quantitative pipeline operates across seven deterministic stages:

### 1. Regulatory Ingestion (`fetch_text.py`)
Polls the SEC EDGAR Submissions API for monitored mega cap issuers (e.g., AAPL, MSFT, NVDA, TSLA, GOOGL, META, AMZN, JPM, XOM, WMT). Raw disclosures are sanitized by stripping uninformative HTML markup, scripts, and tabular exhibits to isolate narrative executive prose.

### 2. Financial Sentiment and Contextual Extraction (`analyze_tone.py`)
Text is partitioned into sentence chunks $C = \{c_1, c_2, \dots, c_n\}$ and classified via **FinBERT** (`ProsusAI/finbert`). Let $L(c_i) \in \{\text{positive}, \text{negative}, \text{neutral}\}$ denote the assigned polarity. Aggregate tone is computed as:

$$\text{Tone} = \frac{1}{|C|} \sum_{i=1}^{|C|} \mathbf{1}_{\{L(c_i) = \text{positive}\}} - \frac{1}{|C|} \sum_{i=1}^{|C|} \mathbf{1}_{\{L(c_i) = \text{negative}\}} = \frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{total}}}$$

where $\text{Tone} \in [-1.0, 1.0]$. To preserve qualitative interpretability, the pipeline extracts the most salient negative chunk $c^*$ defined by maximum negative posterior confidence:

$$c^* = \arg\max_{c_i \in C, \, L(c_i) = \text{negative}} P(\text{negative} \mid c_i)$$

The excerpt is bounded to 150 characters to furnish immediate contextual evidence (`worst_snippet`).

### 3. Narrative Semantic Drift (`drift.py`)
Each document $D$ is embedded into a dense latent representation $\mathbf{v} \in \mathbb{R}^{384}$ using **Sentence Transformers** (`all-MiniLM-L6-v2`) via mean pooling across sentence chunks:

$$\mathbf{v} = \frac{1}{|C|} \sum_{k=1}^{|C|} \text{Encoder}(c_k)$$

The semantic narrative drift between consecutive filings at periods $t$ and $t-1$ is quantified by the cosine distance:

$$\text{Drift}(t, t-1) = 1 - \frac{\mathbf{v}_t \cdot \mathbf{v}_{t-1}}{\|\mathbf{v}_t\|_2 \|\mathbf{v}_{t-1}\|_2}$$

where $\text{Drift} \in [0.0, 2.0]$. Values $\ge 0.50$ indicate thematic divergence or restructuring in corporate disclosures.

### 4. Statistical Anomaly Detection (`anomalies.py`)
For an issuer historical longitudinal tone series $\mathcal{T} = \{\text{Tone}_t\}_{t=1}^T$, we compute the empirical mean $\mu$ and standard deviation $\sigma$:

$$\mu = \frac{1}{T} \sum_{t=1}^T \text{Tone}_t, \quad \sigma = \sqrt{\frac{1}{T} \sum_{t=1}^T (\text{Tone}_t - \mu)^2}$$

The standardized score $z_t$ is defined as:

$$z_t = \frac{\text{Tone}_t - \mu}{\sigma + \epsilon}$$

An 8K disclosure is classified as an anomalous event under the quantitative decision boundary:

$$\text{Anomaly}(t) = \begin{cases} 1 & \text{if } |z_t| \ge 1.50 \;\;\lor\;\; \text{Drift}(t, t-1) \ge 0.50 \quad (\text{with } T \ge 3) \\ 0 & \text{otherwise} \end{cases}$$

### 5. Financial Ground Truth Validation (`yfinance`)
To prevent evaluating natural language signals in an empirical vacuum, each detected anomaly is cross referenced with equity market ground truth using `yfinance`. The system computes the simple annual percentage return for the filing year:

$$\text{Return}_t = \left(\frac{P_{t, \text{final}} - P_{t, \text{initial}}}{P_{t, \text{initial}}}\right) \times 100$$

This metric correlates whether an anomalous negative or defensive executive tone was reflected in equity price depreciation or whether the shock was absorbed by the market. Non blocking exception boundaries gracefully handle API rate limits or symbol delistings.

### 6. Idempotent Real Time Event Dispatch (`notify.py`)
When $\text{Anomaly}(t) = 1$, the event dispatcher constructs an institutional grade alert payload:

```text
SEC 8K Tone Anomaly Alert

Ticker: AAPL (2026)
Z Score: +1.71 sigma | Drift: 0.360
Annual Equity Return: +28.71%

Key Negative Excerpt:
> "Form 8K disclosure analysis complete (Item 8.01 Other Events)..."
```

To ensure strict idempotence across continuous pipeline executions, notified events are committed to a deduplication ledger $\mathcal{K} \subset \text{Issuer} \times \text{Year}$ serialized at `data/state/notified.json`.

### 7. Institutional Quantitative Web Terminal (`dashboard.py` / `docs/`)
The system compiles a standalone Bloomberg and FactSet style HTML dashboard with Plotly.js charts:
- **Longitudinal Tone Z Score Trajectory:** Standardized deviation plotted against upper and lower critical risk bands ($\pm 1.50\sigma$).
- **2D Risk Dispersion (Tone Z vs. Semantic Drift):** Identifies securities in Quadrant IV (*Compound Tail Risk*).
- **Live Exceedances Table:** Real time table detailing period, raw FinBERT tone, drift, equity 1D annual return, and interactive modal disclosure inspection.

### 8. Autonomous GitOps Persistence (`.github/workflows/analyze.yml`)
A scheduled GitHub Actions runner executes the analysis daily (configured at 02:23 UTC to avoid peak queue latency following EDGAR closure). Validated model outputs, state ledgers, interactive dashboards, and figures are versioned directly into the repository by `github-actions[bot]`.

---

## Repository Structure

```
filing-tone/
├── .github/workflows/
│   ├── analyze.yml          # Daily 24h cron execution and autonomous bot commit
│   └── ci.yml               # Automated test validation on push/PR with pip caching
├── docs/
│   └── index.html           # Institutional Bloomberg style interactive dashboard
├── scripts/
│   ├── run_all.py           # Pipeline orchestrator
│   ├── common.py            # Decision boundaries, sentence chunking, and statistics
│   ├── fetch_text.py        # SEC EDGAR ingestion module
│   ├── analyze_tone.py      # FinBERT sentiment classification and context extraction
│   ├── drift.py             # Sentence Transformers semantic drift evaluation
│   ├── anomalies.py         # Anomaly detection and yfinance market ground truth
│   ├── dashboard.py         # Plotly serializer and dashboard HTML compiler
│   ├── notify.py            # Telegram alert dispatcher with deduplication
│   ├── templates/           # Bloomberg UI HTML and CSS templates
│   ├── data/
│   │   ├── scores/          # Serialized filing score metrics (JSON)
│   │   └── state/           # Deduplication ledger (notified.json)
│   └── measurements/        # Anomaly summaries and generated figures
├── tests/
│   └── test_common.py       # Unit test suite (decision boundary, ground truth, notify)
├── .env.example             # Template for Telegram Bot credentials
└── requirements.txt         # Project dependencies (FinBERT, SBERT, yfinance, PyTorch)
```

---

## Experimental Replication

### Environment Setup
```bash
git clone https://github.com/JoelDev13/filing-tone.git
cd filing-tone
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Optional: Configure real time Telegram alerts
cp .env.example .env
# Set TELEGRAM_TOKEN and TELEGRAM_CHAT in .env
```

### Running the Test Suite
```bash
pytest -v
```

### Executing the Pipeline Manually
```bash
python scripts/run_all.py
```

---

## References

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pretrained Language Models*. arXiv preprint [arXiv:1908.10063](https://arxiv.org/abs/1908.10063).
- Reimers, N., and Gurevych, I. (2019). *Sentence BERT: Sentence Embeddings using Siamese BERT Networks*. In Proceedings of EMNLP [arXiv:1908.10084](https://arxiv.org/abs/1908.10084).
