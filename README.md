# HealthPulse

**A longitudinal biomarker trend dashboard that turns repeated urine test results into personalized, explainable health signals.**

Most diagnostic tools answer one question: *"Is this value normal right now?"* HealthPulse answers a more useful one: *"Is this changing in a way that matters for **this person**, based on **their own** history?"*

---
🔗 **Live demo:** [https://healthpulse-fbeuszoaceghgddgxqs9y5.streamlit.app/](https://healthpulse-fbeuszoaceghgddgxqs9y5.streamlit.app/)
## Why this exists

Population reference ranges are a blunt instrument. A creatinine level that's alarming for one person can be perfectly normal for another. The clinically meaningful signal is often not the absolute value it's the *trajectory*: is this person drifting away from their own baseline, and how fast?

HealthPulse is an end-to-end system that:
1. Models what realistic longitudinal urine-biomarker data looks like across six clinical archetypes, grounded in published reference ranges
2. Learns to flag risk and anomalies from that data using two complementary ML approaches
3. Tracks each person against **their own personal baseline**, not just population thresholds
4. Surfaces all of this through an interactive dashboard a non-technical user could actually read

---

## What it does

- **Tracks 5 biomarkers over time** per user: creatinine, albumin, glucose, pH, specific gravity plus a computed uACR (urine albumin-to-creatinine ratio)
- **Stages kidney risk** using KDIGO 2022 clinical thresholds (normal / microalbuminuria / macroalbuminuria)
- **Flags anomalies** with an unsupervised Isolation Forest — catching outlier test results without ever being told what "abnormal" means
- **Predicts risk tier** with a supervised Random Forest classifier trained on biomarker patterns
- **Scores deviation from personal baseline** using z-scores computed against each user's own first two months of data, not a population average
- **Detects transient spikes** with a 3-month sliding window comparison, catching issues that fully resolve before the next test
- **Generates plain-language alerts** ("Albumin steadily rising months 10–12, may indicate early kidney stress. Monitor monthly.") rather than raw numbers
- **Live prediction tool** enter new biomarker values and see the risk classification, anomaly flag, and KDIGO stage update in real time, with a confidence-broken-down probability gauge

---

## A deliberate design decision worth knowing about

The Random Forest classifier is trained **without uACR as an input feature**, even though uACR is exactly what would make the model's job trivial. Why exclude the single most predictive variable?

Because uACR is what the *label itself* is derived from. Feeding it into the model as a feature wouldn't be prediction it would be the model reading the answer off the same page the question was written on. So the classifier instead has to infer risk from creatinine, albumin, glucose, pH, and specific gravity alone the way a real screening tool would have to work if it were trying to catch risk *before* someone has an official diagnostic reading.

One visible consequence: the model's risk-tier prediction and the rule-based KDIGO check don't always agree on the exact same case. That's not a bug it's the honest cost of avoiding leakage, and it's a more meaningful result than a model that simply memorized its own label.

---

## Results

Trained and evaluated on a 6,000-row synthetic dataset (500 users × 12 months, across 6 clinical archetypes see [Data](#data--its-limits) below).

**Random Forest risk classifier** — 5-fold CV accuracy: **0.989 ± 0.005**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Low Risk | 0.99 | 1.00 | 0.99 |
| Medium Risk | 0.98 | 0.95 | 0.96 |
| High Risk | 1.00 | 0.75 | 0.86 |

Feature importance: **albumin (51%)** dominates, followed by creatinine (17%), pH (14%), specific gravity (13%), glucose (6%) — consistent with clinical expectations for kidney-risk screening.

Learning curves show a train/validation gap of ~0.02, with no evidence of overfitting.

**Isolation Forest anomaly detection** flags outlier test results at meaningfully different rates across conditions: 91% of moderate-CKD tests flagged as anomalous vs. just 3% of healthy-user tests showing the model is picking up genuine signal, not noise.

---

## Data — and its limits

No public longitudinal urine-wellness dataset exists, so the dataset is synthetically generated but not arbitrarily. Each of the six condition archetypes (healthy, early CKD, moderate CKD, diabetic risk, CKD + diabetes, UTI episode) has biomarker trajectories grounded in published clinical reference ranges, with realistic within-person variance and condition-appropriate progression over 12 months.

**This is a deliberate, disclosed limitation, not an oversight.** Synthetic data lets the pipeline demonstrate correct methodology leakage avoidance, personal baselining, proper cross-validation — but the resulting accuracy numbers describe how well the models recover a rule-based synthetic label, not how they'd perform on real-world variability. Validating against real biomarker data is the natural next step before any of this could be trusted in a real screening context.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data generation & modeling | Python, NumPy, pandas, scikit-learn |
| Models | Random Forest (risk classification), Isolation Forest (anomaly detection), StandardScaler |
| Dashboard | Streamlit, Plotly |
| Analysis / experimentation | Jupyter Notebook |

---

## Project structure

```
HealthPulse/
├── app.py                      # Streamlit dashboard (4 pages)
├── HealthPulse.ipynb           # Data generation, EDA, modeling, evaluation
├── requirements.txt
├── healthpulse_dataset.csv     # Full 6,000-row synthetic dataset
├── user_summary.csv            # Per-user trend & baseline summary
├── models/
│   ├── risk_classifier.pkl
│   ├── isolation_forest.pkl
│   └── scaler.pkl
├── charts/                     # Saved evaluation & EDA figures
└── assets/                     # Dashboard icons
```

## Dashboard pages

1. **Dashboard** at-a-glance metrics, trend sparklines, latest alert, and a 12-month uACR chart with personal baseline band overlaid
2. **Biomarker Trends** deep dive into any single biomarker, with threshold lines, trend slope, and personal z-score chart
3. **Alerts** full alert history and anomaly-rate breakdown across all users
4. **Live Prediction** enter new biomarker values and get an instant risk classification, anomaly check, and KDIGO stage, with a probability-confidence gauge

---

## Running it locally

```bash
# 1. Clone / unzip the project, then from the project root:
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.

To regenerate the dataset, retrain the models, or inspect the full analysis, open `HealthPulse.ipynb`.

---

## What I'd do next with real data

- Validate the risk classifier and anomaly detector against real longitudinal biomarker readings, not synthetic labels
- Re-calibrate the Isolation Forest's contamination parameter against real-world anomaly prevalence, rather than the known synthetic rate
- Extend the personal-baseline approach to handle users with very short histories (fewer than 2–3 months of data), where a personal baseline can't yet be reliably estimated
- Add confidence intervals to trend slopes, so a "rising" trend is only flagged once it's statistically distinguishable from noise

---

## A note on how this project was built

Every modeling choice here — the leakage exclusion, the personal-baseline approach over population thresholds, the sliding-window anomaly detection — was a deliberate design decision, not a default. Where the pipeline has real limits (synthetic data, a rule-based ground truth), they're called out directly rather than glossed over. I'd rather show accurate, well-reasoned work with disclosed limitations than inflated numbers without context.
