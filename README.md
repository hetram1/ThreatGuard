# 🛡️ ThreatGuard

### Real-Time AI Digital Threat Intelligence Platform

[![Status](https://img.shields.io/badge/status-active%20prototype-0e7490?style=for-the-badge)](https://github.com/hetram1/ThreatGuard)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=flat-square&logo=react&logoColor=111827)](https://react.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Persistence-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Vite](https://img.shields.io/badge/Vite-Tooling-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vite.dev/)

> **ThreatGuard turns suspicious digital content into a structured, explainable risk assessment — without visiting the destination.**

ThreatGuard is an end-to-end AI/ML cybersecurity application for detecting phishing URLs and scam or smishing messages, translating model output into deterministic risk bands, persisting the analysis trail, and presenting the workflow through a responsive security dashboard.

It combines supervised learning, feature engineering, model-contract validation, risk interpretation, FastAPI, PostgreSQL, React, and browser-based QR ingestion into one coherent application.

---

## ⚡ At a Glance

| Capability | Implementation |
|---|---|
| 🔗 URL threat detection | Character TF-IDF + 45 engineered URL features + Logistic Regression |
| 💬 SMS scam detection | Word TF-IDF + Character TF-IDF + Logistic Regression |
| 📷 QR camera scanning | Browser QR decoding → URL detector |
| 🖼️ QR image scanning | Image decoding → URL detector |
| 🎯 Risk scoring | LOW / MEDIUM / HIGH / CRITICAL |
| 🧾 Explainability | URL structural signals + SMS heuristic indicators |
| 🚀 API | FastAPI + Pydantic |
| 💾 Persistence | PostgreSQL + SQLAlchemy |
| 🖥️ Dashboard | React + Recharts + Lucide |
| 🧪 Auditability | Metric JSON + manifests + audit scripts |

---

# 🎯 Problem

Threats reach users through multiple entry points:

~~~text
Suspicious URL ───────────────┐
Suspicious SMS ───────────────┼──→ AI Detection
QR Camera ──→ URL ────────────┤          ↓
QR Image ───→ URL ────────────┘     Risk Engine
                                        ↓
                                      FastAPI
                                        ↓
                                   PostgreSQL
                                        ↓
                                 React Dashboard
~~~

A standalone classifier is not enough. ThreatGuard adds the surrounding application engineering needed to turn model inference into a usable security workflow.

---

# 🧠 System Architecture

## High-Level

~~~mermaid
flowchart LR
    A[URL Input] --> U[URL Detector]
    B[SMS / Message] --> S[SMS Detector]
    C[QR Camera] --> Q[QR Decoder]
    D[QR Image] --> Q

    Q -->|HTTP / HTTPS URL| U

    U --> R[Risk Engine]
    S --> R

    R --> T[Unified Threat Analysis Service]
    T --> F[FastAPI]

    F --> P[(PostgreSQL)]
    F --> W[React Dashboard]

    W --> H[History + Metrics + Risk Visualization]
~~~

## Request Lifecycle

~~~mermaid
sequenceDiagram
    participant User
    participant UI as React Dashboard
    participant API as FastAPI
    participant Engine as Risk Engine
    participant Model as ML Model
    participant DB as PostgreSQL

    User->>UI: Submit URL / SMS / QR
    UI->>API: Analysis request
    API->>Engine: Normalize + analyze
    Engine->>Model: Predict probability
    Model-->>Engine: Probability + class
    Engine->>Engine: Apply risk policy + signals
    Engine->>API: Structured result
    API->>DB: Persist result
    API-->>UI: Score + band + reasons
    UI-->>User: Threat assessment
~~~

### 🔐 QR security boundary

~~~text
QR image / camera
       ↓
Decode payload
       ↓
Accept HTTP/HTTPS URL
       ↓
Run URL analysis
       ↓
Never open the destination
~~~

The QR scanner is an ingestion channel, not a browsing mechanism.

---

# 🔬 Machine Learning

ThreatGuard intentionally uses separate models for URLs and messages because their statistical structures are different.

## 1. URL Phishing Detector — V5

### Production pipeline

~~~text
URL
 ↓
Character TF-IDF
  n-grams: 3–5
  max features: 150,000
 ↓
45 engineered URL features
 ↓
Standard scaling
 ↓
Feature concatenation
 ↓
Logistic Regression
 ↓
Phishing probability
~~~

### Engineered feature families

The production feature extractor captures:

- URL, hostname and path lengths
- digit, letter and special-character ratios
- dots, hyphens, slashes and separators
- subdomain and path depth
- IP-based hosts
- explicit ports
- embedded user information
- percent and hexadecimal encoding
- suspicious keywords
- entropy and token structure
- repeated path separators
- suspicious TLD indicators

### V5 training corpus

| Source | Rows |
|---|---:|
| UCI PhiUSIIL | 235,370 |
| Basic benign augmentation | 100,000 |
| Deep-path benign augmentation | 200,000 |
| **Combined** | **535,370** |

### V5 holdout metrics

| Metric | Score |
|---|---:|
| Accuracy | **99.8412%** |
| Precision | **99.9549%** |
| Recall | **99.1992%** |
| F1 | **99.5756%** |
| ROC-AUC | **99.9346%** |
| PR-AUC | **99.8582%** |

Evidence: [URL V5 metrics](models/url_v5_metrics.json)

### Unseen-domain audit

A separate repository audit evaluates domains not seen during training.

| Audit measure | Value |
|---|---:|
| Unseen domains | **43,819** |
| Test URLs | **47,005** |
| Accuracy | **99.8872%** |
| Precision | **99.9950%** |
| Recall | **99.7403%** |
| F1 | **99.8675%** |
| ROC-AUC | **99.9800%** |
| PR-AUC | **99.9825%** |
| False positives | **1** |
| False negatives | **52** |

Evidence: [URL detector manifest](models/url_detector_manifest.json)

> The audit is evidence about the evaluated sample, not a guarantee about arbitrary future URLs. ThreatGuard should be treated as a risk signal rather than proof that a URL is safe or malicious.

---

## 2. SMS Scam Detector — V1

### Production pipeline

~~~text
SMS / Message
   ↓
Word TF-IDF
  n-grams: 1–2
  max features: 120,000
   +
Character TF-IDF
  n-grams: 3–5
  max features: 120,000
   ↓
Feature concatenation
   ↓
Logistic Regression
  C = 4
  class_weight = balanced
  solver = liblinear
   ↓
Scam probability
~~~

### Dataset

Production manifest: **sidzzz07/scamshield-dataset**

Exact duplicate texts are removed within each preserved split.

| Split | Samples |
|---|---:|
| Train | 69,237 |
| Validation | 8,191 |
| Test | 8,201 |

### Test metrics

| Metric | Score |
|---|---:|
| Accuracy | **97.8295%** |
| Precision | **96.8884%** |
| Recall | **97.2102%** |
| F1 | **97.0491%** |
| ROC-AUC | **99.6449%** |
| PR-AUC | **99.4378%** |
| False positives | **94** |
| False negatives | **84** |

Evidence: [SMS detector manifest](models/sms_detector_manifest.json)

### Threshold

Production classification threshold: **0.50**

Threshold analysis is retained in the model artifacts. The displayed score is a model-derived risk signal, not a calibrated probability estimate.

---

# 🚦 Risk Engine

Model probability is translated into deterministic application-level severity:

| Probability | Risk band |
|---:|---|
| below 0.20 | 🟢 LOW |
| 0.20 to below 0.50 | 🟡 MEDIUM |
| 0.50 to below 0.80 | 🟠 HIGH |
| 0.80 and above | 🔴 CRITICAL |

The risk layer does not retrain or modify the classifier. It interprets model output for application use.

---

# 🧩 Explainability

ThreatGuard keeps model inference separate from supporting heuristic signals.

### URL evidence

The URL engine can surface indicators such as:

- IP address instead of hostname
- non-default port
- embedded user information
- deep subdomain structure
- unusually long host or URL
- encoded or hexadecimal patterns
- suspicious keywords
- suspicious TLD indicators
- repeated path separators

### SMS evidence

The SMS engine can surface:

- security or account language
- urgency or time pressure
- reward or prize language
- payment or money language
- web links

These are explanatory indicators, not causal proof of a model decision.

The repository also includes offline SHAP-based model-audit tooling.

---

# 🏗️ Backend Engineering

The backend is deliberately separated into contracts, orchestration, inference, and persistence:

~~~text
FastAPI
  │
  ├── Pydantic contracts
  │
  ├── ThreatAnalysisService
  │      ├── URLRiskEngine
  │      └── SMSRiskEngine
  │
  └── AnalysisRepository
         ↓
      SQLAlchemy
         ↓
      PostgreSQL
~~~

### Engineering decisions

**Thin API layer**  
Routes validate input, delegate analysis, persist results, and return a stable response model.

**Unified inference boundary**  
ThreatAnalysisService gives the application a common interface while preserving independent URL and SMS engines.

**Feature contract validation**  
Production engines check that trained coefficients, vectorizers, scalers, and feature definitions agree before inference.

**Persistent audit trail**  
Every analysis stores input type, input value, prediction, score, risk band, confidence, model metadata, reasons, and timestamp.

---

# 💾 PostgreSQL Persistence

The core analysis entity is:

~~~text
analyses
├── id
├── input_type
├── input_value
├── prediction
├── risk_score
├── risk_band
├── confidence
├── model_name
├── model_version
├── reasons
└── created_at
~~~

This enables recent history, aggregate dashboard views, model/version visibility, and timestamped analysis records.

---

# 🖥️ React Security Dashboard

A compact SOC-style interface is provided for interactive triage.

| Area | Capability |
|---|---|
| Analysis | URL + SMS |
| QR | Camera + image upload |
| Results | Prediction + score + confidence |
| Explainability | Human-readable reasons |
| Analytics | Threat + risk distributions |
| Persistence | PostgreSQL-backed history |
| Health | API online/offline state |
| UX | Responsive dark security UI |

### QR scanner flow

The browser uses **html5-qrcode** to:

1. open the camera or select an image,
2. decode QR content,
3. accept HTTP/HTTPS destinations,
4. route the decoded URL into the existing URL detector.

No separate QR ML classifier is required.

---

# 🔌 API

Base URL:

~~~text
http://localhost:8000
~~~

Interactive OpenAPI documentation:

~~~text
http://localhost:8000/docs
~~~

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /api/v1/health | Service health |
| POST | /api/v1/analyze/url | URL analysis |
| POST | /api/v1/analyze/sms | SMS analysis |
| POST | /api/v1/analyze | Unified URL/SMS analysis |
| GET | /api/v1/model-info | Model metadata |
| GET | /api/v1/history | Recent persisted analyses |

### URL request

~~~json
{
  "url": "https://example.com/account/verify"
}
~~~

### SMS request

~~~json
{
  "text": "URGENT! Your account will be suspended. Verify KYC immediately."
}
~~~

### Unified request

~~~json
{
  "input_type": "url",
  "value": "https://example.com/account/verify"
}
~~~

---

# 🗂️ Repository Map

~~~text
ThreatGuard/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── database/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── api.js
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── ml/
│   ├── sms_detection/
│   └── url_detection/
│
├── models/
│   ├── manifests
│   ├── metrics
│   └── calibration artifacts
│
├── data/
├── docker-compose.yml
├── requirements.txt
├── requirements-lock.txt
└── README.md
~~~

Small JSON evaluation artifacts are tracked; generated datasets and large model binaries are intentionally ignored.

---

# 🚀 Run Locally

## 1. Clone

~~~bash
git clone https://github.com/hetram1/ThreatGuard.git
cd ThreatGuard
~~~

## 2. Python environment

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

## 3. PostgreSQL

~~~bash
docker compose up -d postgres
docker compose ps
~~~

## 4. Backend

~~~bash
source .venv/bin/activate
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
~~~

Open:

~~~text
http://localhost:8000
http://localhost:8000/docs
~~~

## 5. Frontend

In another terminal:

~~~bash
cd frontend
npm install
npm run dev
~~~

Optional environment file:

~~~bash
cp .env.example .env
~~~

Default API target:

~~~text
VITE_API_BASE_URL=http://localhost:8000
~~~

---

# ✅ Validation

The current application has been exercised across the full workflow:

~~~text
URL
✓ legitimate URL
✓ suspicious / phishing URL

SMS
✓ legitimate message
✓ scam message

Unified API
✓ URL dispatch
✓ SMS dispatch

QR
✓ camera decode
✓ device-image decode
✓ decoded URL → URL detector

Application
✓ PostgreSQL persistence
✓ history retrieval
✓ dashboard rendering
✓ frontend lint
✓ production build
~~~

Basic local checks:

~~~bash
curl http://localhost:8000/api/v1/health

cd frontend
npm run lint
npm run build
~~~

---

# 🔐 Security & Design Notes

### Risk signal ≠ proof

A classifier can make mistakes. ThreatGuard is intended to provide an evidence-backed risk signal, not a definitive security guarantee.

### No live destination browsing

The production URL engine performs lexical and structural inference. It does not need to open the destination website.

### QR does not auto-navigate

QR payloads are decoded and, when they contain HTTP/HTTPS URLs, routed into URL analysis. The scanner does not automatically browse to them.

### Explanations are separate from inference

Reason strings come from deterministic heuristic checks and are returned as supporting indicators rather than as the model's causal explanation.

### Artifact discipline

- Raw and generated datasets are ignored.
- Large binary model artifacts are ignored.
- Metric, manifest, and calibration JSON files remain tracked.
- Environment files are ignored except for example configuration.

---

# 🧪 Engineering Questions This Repository Can Answer

**Why character n-grams for URLs?**  
URLs are structurally different from natural language. Character features capture delimiters, path fragments, spelling variants, encoding patterns, and unusual token composition.

**Why combine character features with engineered URL features?**  
The learned representation captures distributed lexical patterns while engineered features explicitly represent host, path, encoding, depth, separator, and composition signals.

**Why separate risk policy from the model?**  
The classifier owns inference. The application layer owns deterministic mapping from model output to operational severity and user-facing signals.

**Why PostgreSQL?**  
Analyses are application events. Persistence creates an audit trail and enables history and aggregate views.

**Why a unified inference service?**  
It keeps API routes thin while preserving modality-specific models behind one application contract.

**Why reuse URL detection for QR?**  
A QR code is a transport format. After decoding, the security-relevant artifact is the URL itself.

**Why keep evaluation artifacts in JSON?**  
Model evaluation should be inspectable without rerunning notebooks. Small manifests and metric files provide a lightweight experiment record.

---

# 📈 Future Extension Points

~~~text
Current
URL / SMS / QR
      ↓
Threat Analysis
      ↓
PostgreSQL
      ↓
Dashboard

Extension points
├── authentication / RBAC
├── rate limiting
├── analyst feedback
├── alerting
├── richer threat feeds
├── model drift monitoring
├── calibration tracking
└── CI/CD + deployment automation
~~~

These are extension points rather than requirements of the current prototype.

---

# 📚 Model Evidence

Core model artifacts:

- [URL V5 metrics](models/url_v5_metrics.json)
- [URL detector manifest](models/url_detector_manifest.json)
- [SMS detector manifest](models/sms_detector_manifest.json)
- [SMS baseline metrics](models/sms_baseline_metrics.json)
- [SMS threshold calibration](models/sms_threshold_calibration.json)

The repository therefore exposes both the application implementation and the evaluation evidence behind its deployed inference behavior.

---

# 🛠️ Technology Stack

### Machine Learning
**Python · NumPy · Pandas · scikit-learn · XGBoost · SHAP · SciPy · Joblib**

### Backend
**FastAPI · Pydantic · SQLAlchemy · PostgreSQL · psycopg2 · Uvicorn**

### Frontend
**React · Vite · Recharts · Lucide React · html5-qrcode · CSS**

### Development / Data
**Docker Compose · Jupyter · audit scripts · JSON evaluation manifests**

---

# 👨‍💻 Project Philosophy

ThreatGuard is built around one engineering question:

> **Can suspicious digital content be transformed into a fast, explainable, auditable risk decision without opening the underlying destination?**

The implementation spans:

~~~text
Dataset
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Evaluation / Audit
   ↓
Production Risk Engine
   ↓
Unified Inference Service
   ↓
FastAPI
   ↓
PostgreSQL
   ↓
React Dashboard
   ↓
Human-facing Threat Assessment
~~~

---

## ⭐ Reviewer / Interviewer Path

For a rapid technical review:

**1. ML pipeline** → [ml/](ml/)  
**2. Production inference** → [backend/app/services/](backend/app/services/)  
**3. API layer** → [backend/app/main.py](backend/app/main.py)  
**4. Persistence** → [backend/app/models/](backend/app/models/)  
**5. Frontend** → [frontend/src/](frontend/src/)  
**6. Model evidence** → [models/](models/)

That order follows the same conceptual chain as the running system:

**data → features → model → risk engine → API → database → UI**

---

<div align="center">

### 🛡️ ThreatGuard

**AI-assisted threat detection · Explainable risk signals · No destination browsing**

[View Repository](https://github.com/hetram1/ThreatGuard)

</div>
