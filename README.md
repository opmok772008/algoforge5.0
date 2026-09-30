# Streaming ECG Arrhythmia Analysis Demo

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-00e5a3?style=for-the-badge&logo=github)](https://opmok772008.github.io/algoforge4.0/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Uvicorn-009688?style=for-the-badge&logo=fastapi)](http://localhost:8000/docs)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)

A multi-file Python/FastAPI research demo for noise-resilient ECG analysis. It generates deterministic normal, ventricular tachycardia (VT), and ventricular fibrillation (VF) signals; removes baseline wander; detects R-peaks; estimates RR/HRV features and signal quality; applies fuzzy alarm/hold rules; and tunes detector parameters with a genetic algorithm. The website also exposes a streaming simulator, ECG upload and analysis, optimization, health, and validation endpoints.

The built-in records are synthetic, and uploaded traces are analyzed with heuristic signal processing. This project is an educational prototype, not a clinical monitor or medical device; its demo results are not clinical performance claims.

---

## 🎯 Problem Context and Demo Scope

> **Problem context**: ICU monitors can raise false alarms from noisy ECG (baseline wander, muscle and motion artifacts), contributing to alarm fatigue. This project demonstrates a synthetic signal-processing workflow that explores detection and noise-aware alarm logic.
>
> **Method**:
> - **Clean**: Moving-average baseline removal preserves QRS amplitude ($\ge 85\%$).
> - **Detect**: An adaptive-threshold detector with a 200 ms refractory period finds R-peaks.
> - **Judge**: Fuzzy rules combine heart rate, rhythm regularity and signal quality into an alarm confidence. Noise-like input is held for review instead of alarming.
> - **Optimize**: A genetic algorithm tunes four detector settings (`bw`, `sm`, `th`, `mw`) on three noisy records. Fitness is F1 score minus an R-peak jitter penalty, with elitism.
>
> **Validation**: The built-in synthetic validation endpoint checks detector sensitivity and precision on generated clean rhythm, timing under baseline wander, QRS amplitude, VT/VF demo alert timing, EMG noise handling, fuzzy confidence monotonicity, fixed streaming-buffer size, and genetic-algorithm elitism. These checks describe simulated records only.
>
> **Extras**: RR-feature logic (HR, HRV, RMSSD, NN50) is ported from an open-source ECG repository, and users can upload their own CSV/TXT ECG files.
>
> **Limits**: Built-in signals are synthetic. Uploaded signals are processed with heuristic algorithms. This prototype is not validated for patient care and is not a medical device.

---

## ⚡ Quick Start: Running the Full Website Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Backend Web Server
```bash
python server.py
# or
python run.py
```

### 3. Open in Browser
- **Live Website**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Endpoint**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

*(The frontend also operates in 100% standalone client mode when deployed to GitHub Pages or static hosts).*

### Automated tests and benchmark

Run the regression suite and the repeatable latency/memory benchmark locally:

```bash
python -m pytest -q
python benchmark.py
```

The benchmark generates synthetic records at several lengths and reports median and p95 analysis time, throughput, and Python/NumPy traced peak allocation. It excludes signal-generation time and is not a clinical performance benchmark.

---

## 🏛️ System Architecture

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │                 RAW ECG SENSOR / FILE INPUT                 │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                                                 ▼
                  ┌─────────────────────────────────────────────────────────────┐
                  │ 1. CLEAN: Moving-Average Baseline Removal (Window: bw)      │
                  │    - Filters respiration & perspiration baseline wander     │
                  │    - Preserves >= 85% of true 1.0 mV QRS peak amplitude     │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                                                 ▼
                  ┌─────────────────────────────────────────────────────────────┐
                  │ 2. DETECT: Adaptive Threshold + 200 ms Refractory Period    │
                  │    - Smoothing (sm) -> Differentiation & Squaring           │
                  │    - Moving Window Integration (mw)                         │
                  │    - Adaptive Threshold: max(th * P97, 2.5 * P50)           │
                  │    - 200 ms Blanking prevents double-counts & T-wave errors │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                                                 ▼
                  ┌─────────────────────────────────────────────────────────────┐
                  │ 3. JUDGE: Fuzzy Logic Inference & Safety Gate               │
                  │    - Inputs: Heart Rate (HR), Regularity (CV), SQI Quality  │
                  │    - Output: Confidence & Noise Suppression                 │
                  │    - Alarm: Lethal VT/VF alert within 3.0 s speed budget    │
                  │    - Hold: Routes noise to "Hold for Review" (0 false alarms)
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                                                 ▼
                  ┌─────────────────────────────────────────────────────────────┐
                  │ 4. OPTIMIZE: Genetic Algorithm with Elitism                 │
                  │    - Optimizes [bw, sm, th, mw] across 3 noisy records       │
                  │    - Fitness = mean( F1 - 0.005 * jitter_ms )               │
                  │    - Elitism preserves best solution across generations     │
                  └─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Automated Validation Test Results

Run via the UI or `GET /api/validation`:

| # | Validation Constraint | Target Threshold | Measured Result | Status |
|---|---|---|---|:---:|
| **V1** | Clean Rhythm Sensitivity & Precision | $\ge 95\%$ | **Se 100.0%, PPV 100.0%** | **PASS** |
| **V2** | R-peak Jitter under Baseline Wander | $\le 20\text{ ms}$ | **2.6 ms** | **PASS** |
| **V3** | QRS Amplitude Preservation | $\ge 85\%$ of 1.0 mV | **0.89 mV (88.9%)** | **PASS** |
| **V4a**| Ventricular Tachycardia (VT) Alert Latency | $\le 3.0\text{ s}$ | **2.4 s** | **PASS** |
| **V4b**| Ventricular Fibrillation (VF) Alert Latency| $\le 3.0\text{ s}$ | **2.4 s** | **PASS** |
| **V5** | Severe EMG Burst False Alarm Rejection | $0\text{ alarms in }20\text{ s}$ | **0 false alarms** | **PASS** |
| **V6** | Monotonic Fuzzy Confidence with Heart Rate | Non-decreasing | **0.00 0.50 1.00 1.00 1.00** | **PASS** |
| **V7** | Circular Streaming Buffer Memory | Fixed 6 KB (1500 samples)| **6000 bytes** | **PASS** |
| **V8** | Elitism Convergence Stability | Non-decreasing fitness | **0.930 to 0.973** | **PASS** |

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive web application |
| `GET` | `/api/health` | Service status, active parameters, and buffer metrics |
| `GET` | `/api/params` | Current detector parameters (`bw, sm, th, mw`) |
| `POST` | `/api/params` | Update active detector parameters |
| `POST` | `/api/stream` | Step streaming monitor with rhythm scenario & noise toggles |
| `POST` | `/api/analyze` | Ported open-source RR-feature & HRV metrics extraction |
| `POST` | `/api/upload` | Multipart upload for `.csv`, `.txt`, `.tsv` ECG records |
| `POST` | `/api/optimize` | Launch Genetic Algorithm parameter tuning job |
| `GET` | `/api/optimize/{id}` | Poll generation convergence curve and best parameters |
| `GET` | `/api/validation` | Run all 9 verification checks with measured values |

## Render deployment

The root `render.yaml` defines the FastAPI web service, health check, region, and start command. To create or sync the service, connect this GitHub repository as a Render Blueprint and deploy its `main` branch. After it is linked, commits to `main` trigger a new deploy.

## Suggested submission metadata

**Title:** Streaming ECG Arrhythmia Analysis Demo

**Description:** A Python/FastAPI research demo that processes synthetic ECG streams and uploaded sample traces. It removes baseline wander, detects R-peaks, calculates RR/HRV and signal-quality features, applies fuzzy alarm/hold logic to simulated VT/VF rhythms, and tunes detector settings with a genetic algorithm. The app includes a web dashboard, streaming and analysis endpoints, automated tests, and a reproducible latency/memory benchmark. Built-in data is synthetic; this prototype is not intended for clinical use.

---

## 🌐 Deploy to GitHub & GitHub Pages

To sync any updates to your GitHub repository:
```bash
git add .
git commit -m "Update streaming ECG analysis demo"
git push origin main
```

Your live static website is hosted at:
👉 **[https://opmok772008.github.io/algoforge4.0/](https://opmok772008.github.io/algoforge4.0/)**
