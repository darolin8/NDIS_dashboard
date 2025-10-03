
# NDIS Incident Analytics Dashboard

A Streamlit app for AI-enabled incident analytics tailored to NDIS providers. It ingests incident data, extracts & cleans fields, benchmarks compliance (24-hour / 5-day rules), runs ML for pattern detection and risk insights, and visualises incidents over time and on maps — all in one integrated dashboard.

**Live demo (optional):** if you’ve deployed to Streamlit Cloud, add the link here. For example: `https://zffakcqxacchajyisrczcy.streamlit.app` (replace if different).

## ✨ Key Features

* **Executive overview**: KPIs (incident volume, reportable %, medical attention %), top drivers, and quick wins.
* **Operational performance**: temporal trends (daily/weekly/monthly), cause analysis, carer/role performance, peak-time insights.
* **Compliance & investigation**: automated 24-hour/5-day checks, report delay metrics, investigation triggers, early-risk rules.
* **ML insights**: clustering, correlation, forecasting, classification comparisons, calibrated probabilities, and feature engineering helpers.
* **Interactive map**: geocode incidents, view spatial hotspots, and drill-down by severity/type/time window.
* **Privacy-friendly**: works with synthetic or de-identified data; includes a basic leakage test (`test_leakage.py`).

> Built with Python & Streamlit; code organised into pages, helpers, and utilities for clean iteration. ([GitHub][1])

---

## 📁 Project Structure

```
NDIS_dashboard/
├─ app.py                   # Streamlit app entrypoint (sidebar routing)
├─ dashboard_pages.py       # Page registry + per-page renderers
├─ ml_helpers.py            # ML utilities (features, clustering, models, forecasting)
├─ incident_mapping.py      # Mapping helpers (geocoding + map rendering)
├─ geo_utils.py             # (If present) Spatial helpers
├─ utils/                   # Shared utilities (e.g., factor labels, alerts, theming)
│  ├─ factor_labels.py
│  ├─ alerts.py
│  └─ theme.py
├─ text data/               # Sample/synthetic CSVs or text sources (if provided)
├─ .streamlit/              # Streamlit config (e.g., theme, secrets)
├─ .devcontainer/           # VS Code Dev Containers setup
├─ requirements.txt         # Python deps
└─ test_leakage.py          # Simple privacy/leakage test
```

---

## 🚀 Quick Start

### 1) Clone and set up

```bash
git clone https://github.com/darolin8/NDIS_dashboard.git
cd NDIS_dashboard
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Prepare data

Place your incident dataset (CSV/Parquet) in a folder (e.g., `data/`), with columns similar to:

* `incident_id`
* `incident_date`, `notification_date`
* `incident_type`, `severity`
* `participant_id`, `carer_id`, `role`
* `location` (free-text) or `address/suburb/state/postcode`
* `reportable` (0/1) or `reportable_bin` (0/1)
* `medical_attention` (0/1)
* optional narrative fields, e.g., `description`

> The app computes handy fields like `report_delay_hours` and `within_24h` if `incident_date` and `notification_date` exist.

### 3) Configure (optional but recommended)

**Streamlit secrets** (for API keys, file paths, etc.):

Create `.streamlit/secrets.toml`:

```toml
[general]
data_path = "data/incidents.csv"

[geocoding]
# Choose one; comment the others
opencage_api_key = "YOUR_KEY"
google_maps_api_key = "YOUR_KEY"
```

**App theme**: If you have a custom theme util, it will be applied automatically.

### 4) Run the app

```bash
streamlit run app.py
```

---

## 🧭 How to Use the Dashboard

* **Sidebar**: choose the page (Executive Summary, Operational Performance, Compliance & Investigation, ML Insights, Incident Map).
* **Filters**: apply date windows, severity/incident types, roles, and locations.
* **Tooltips & legends**: hover charts for details; click legends to isolate series.

### Pages at a glance

* **Executive Summary**: KPIs and one-screen health check (volume, reportable %, medical attention %, overdue counts).
* **Operational Performance**: time-series (with seasonality), incident causes by time of day/weekday/month, carer/role benchmarks.
* **Compliance & Investigation**:

  * Computes `report_delay_hours` = `notification_date - incident_date`.
  * Flags 24-hour and 5-day thresholds; shows within-24h rates and overdue counts.
  * Basic investigation rules (e.g., severity, self-harm keywords, late notifications) for early-risk signals.
* **ML Insights**:

  * **Feature engineering**: time lags, locations, roles; optional text vectorisation (TF-IDF/embeddings).
  * **Clustering & correlation**: identify cohorts, drivers, and weak/strong relationships.
  * **Forecasting**: SARIMAX/seasonal decomposition for planning and staffing.
  * **Classification**: compare baseline models; export calibrated probabilities for operations.
* **Incident Map**:

  * Geocoding utilities to convert addresses/suburbs to coordinates.
  * Explore spatial patterns (hotspots) and filter by attributes/time.

---

## 🧪 Testing

Run the leakage test and any added tests:

```bash
pytest -q
```

What the leakage test does:

* Checks that no obviously identifying content or raw PII sneaks into model artifacts or logs.
* Ensures basic guardrails around synthetic/de-identified flows.

---

## 🛠️ Development Notes

### Dev Containers

This repo includes a `.devcontainer/` folder so you can open it directly in VS Code with Docker and get a fully provisioned environment out of the box. ([GitHub][1])

### Coding style

* Keep transformations in helpers (`ml_helpers.py`, `utils/`, `incident_mapping.py`) and rendering in `dashboard_pages.py`.
* Prefer cached data loaders (`@st.cache_data`) for large files.
* Avoid in-place mutation unless necessary; use `.copy()` after rule applications.

---

## 🌏 Geocoding & Maps

You can use:

* **OpenCage** (recommended for ease of use) or
* **Google Maps Geocoding API** (high quality, paid after free tier)

Set your API key in `.streamlit/secrets.toml` and run the geocoder:

```bash
python geocode_addresses.py --in data/addresses.csv --out data/addresses_geocoded.csv
```

(Adjust script/args to match your file; see inline help in the script if present.)

---

## 🧩 Data Expectations (minimum)

At a minimum, the app expects:

* Timestamps that can be parsed (e.g., `incident_date`, `notification_date`).
* Categorical fields like `incident_type`, `severity`, `role`.
* Binary indicators for compliance-relevant flags (e.g., `reportable`, `medical_attention`) if you want those KPIs.

If you don’t have `reportable`, the app will still run; some KPIs will just show `0` / “n/a”.

---

## 🔒 Privacy

* Designed to run on **synthetic or de-identified** data.
* Includes a minimal leakage test and encourages you to keep raw narratives private when sharing screenshots or public demos.

---

## 🚢 Deployment

### Streamlit Community Cloud

1. Push the repo to GitHub (public or private).
2. Create a new Streamlit app, point it to `app.py`, and set Python version + secrets.
3. Add any required environment variables/API keys in the Streamlit secrets UI.

### Docker (example)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build & run:

```bash
docker build -t ndis-dashboard .
docker run -p 8501:8501 ndis-dashboard
```


---

