# Filing Tone: Automated Financial Sentiment & Narrative Drift in SEC 8K Disclosures

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/NLP-FinBERT-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/ProsusAI/finbert)
[![Embeddings](https://img.shields.io/badge/Embeddings-Sentence--Transformers-blue?style=flat-square)](https://sbert.net/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## Abstract

**Filing Tone** is an autonomous, serverless computational framework for longitudinal financial text mining. The system continuously ingests unscheduled material event disclosures (Form 8K) from the U.S. Securities and Exchange Commission (SEC) EDGAR system, extracts sentence level financial sentiment via domain adapted transformer models (FinBERT), and tracks semantic narrative drift over time using dense vector representations. Statistical outliers are identified via z-score thresholds and persisted automatically using a Git as a Datastore paradigm.

---

## Methodology and Mathematical Formulation

The pipeline operates in five deterministic stages:

### 1. Regulatory Ingestion (`fetch_text.py`)
Polls the SEC EDGAR Submissions API for active target issuers (e.g., AAPL, MSFT, NVDA). Documents are sanitized by stripping non-informative HTML markup, scripts, and tabular exhibits to isolate executive prose

### 2. Financial Sentiment Extraction (`analyze_tone.py`)
Text is partitioned into sentence chunks $C = \{c_1, c_2, \dots, c_n\}$ and classified via **FinBERT** (`ProsusAI/finbert`). Let $L(c_i) \in \{\text{positive}, \text{negative}, \text{neutral}\}$ denote the assigned polarity. Aggregate tone is computed as:

$$\text{Tone} = \frac{1}{|C|} \sum_{i=1}^{|C|} \mathbf{1}_{\{L(c_i) = \text{positive}\}} - \frac{1}{|C|} \sum_{i=1}^{|C|} \mathbf{1}_{\{L(c_i) = \text{negative}\}} = \frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{total}}}$$

where $\text{Tone} \in [-1.0, 1.0]$.

### 3. Narrative Semantic Drift (`drift.py`)
Each document $D$ is embedded into a dense latent representation $\mathbf{v} \in \mathbb{R}^{384}$ using **Sentence Transformers** (`all-MiniLM-L6-v2`) via mean pooling across sentence chunks:

$$\mathbf{v} = \frac{1}{|C|} \sum_{k=1}^{|C|} \text{Encoder}(c_k)$$

The semantic narrative drift between consecutive filings at periods $t$ and $t 1$ is quantified by the cosine distance:

$$\text{Drift}(t, t-1) = 1 - \frac{\mathbf{v}_t \cdot \mathbf{v}_{t-1}}{\|\mathbf{v}_t\|_2 \|\mathbf{v}_{t-1}\|_2}$$

where $\text{Drift} \in [0.0, 2.0]$, with higher values indicating radical narrative restructuring

### 4. Statistical Anomaly Detection (`anomalies.py`)
For an issuer's historical longitudinal tone series $\mathcal{T} = \{\text{Tone}_t\}_{t=1}^T$, we compute the empirical mean $\mu$ and standard deviation $\sigma$:

$$\mu = \frac{1}{T} \sum_{t=1}^T \text{Tone}_t, \quad \sigma = \sqrt{\frac{1}{T} \sum_{t=1}^T (\text{Tone}_t - \mu)^2}$$

The standardized score $z_t$ is defined as:

$$z_t = \frac{\text{Tone}_t - \mu}{\sigma + \epsilon}$$

An 8K disclosure is classified as an anomalous event under the decision rule:

$$\text{Anomaly}(t) = \begin{cases} 1 & \text{if } |z_t| \ge 1.5 \;\;\lor\;\; \text{Drift}(t, t-1) \ge 0.5 \\ 0 & \text{otherwise} \end{cases}$$

### 5. Autonomous GitOps Persistence (`.github/workflows/analyze.yml`)
A scheduled GitHub Actions runner executes the analysis every 24 hours. Validated model outputs and figures are versioned directly into the repository by `github-actions[bot]`

---

## Repository Structure

```
filing-tone/
├── .github/workflows/
│   ├── analyze.yml          # Daily 24h cron execution & autonomous bot commit
│   └── ci.yml               # Automated test validation on push/PR
├── scripts/
│   ├── run_all.py           # Pipeline orchestrator
│   ├── common.py            # Sentence chunking and metric definitions
│   ├── fetch_text.py        # SEC EDGAR ingestion module
│   ├── analyze_tone.py      # FinBERT sentiment classification
│   ├── drift.py             # Sentence-BERT semantic drift evaluation
│   ├── anomalies.py         # Statistical anomaly filter & plot generator
│   ├── notify.py            # Event notification handler
│   ├── data/scores/         # Serialized filing score metrics (JSON)
│   └── measurements/        # Anomaly summaries and generated figures
├── tests/
│   └── test_common.py       # Unit test suite
└── requirements.txt         # Environment dependencies
```

---

## Experimental Replication

### Environment Setup
```bash
git clone https://github.com/JoelDev13/filing-tone.git
cd filing-tone
pip install -r requirements.txt
```

---

## References

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. arXiv preprint [arXiv:1908.10063](https://arxiv.org/abs/1908.10063).
.
