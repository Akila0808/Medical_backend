# MedAssist AI - Production FastAPI & MongoDB Backend

A production-ready, highly secure backend for **MedAssist AI**, built with **FastAPI**, **MongoDB (PyMongo / Atlas)**, and **Scikit-Learn**. It provides clinical disease prediction, symptom tracking, patient-doctor management, and medical analytics.

---

## Features

- **FastAPI & Async Architecture**: High-performance RESTful API with automated OpenAPI / Swagger documentation.
- **MongoDB Atlas Integration**:
  - `patients`: Patient accounts with encrypted credentials and demographic metadata.
  - `doctors`: Doctor accounts with specializations and clinical locations.
  - `symptoms_log`: Diagnostic logs linked by email, clinical indicators, predictions, confidence scores, and risk tiers.
  - `appointments`: Patient-doctor appointment booking, statuses (`Pending`, `Accepted`, `Rejected`), and scheduled time slots.
  - Preserves existing datasets: `Sample_diseases`, `Severity`, and `categories`.
- **Role-Based Authentication & Security**:
  - Strict password validation (minimum 8 characters, uppercase, lowercase, numeric digit, and special character).
  - 10-digit phone number validation.
  - Salted password hashing with bcrypt.
  - PyJWT Bearer token authentication (24-hour token expiry).
  - CORS middleware enabled for cross-origin frontend portals (`allow_origins=["*"]`).
- **AI Disease Prediction Engine**:
  - Dynamic model loading for `medassist_disease_model.pkl` and `model_features.pkl` via Joblib.
  - NLP / Fuzzy symptom normalization using `thefuzz`.
  - Clinical safety override checks for acute medical presentations (e.g. cardiac concerns, GI bleeding, severe asthma).
  - Automated risk stratification: **Low**, **Medium**, **High**.
- **Recommendations & Analytics**:
  - Tailored clinical management protocols, precautions, lifestyle advice, and emergency indicators.
  - Aggregation pipeline for prediction statistics, risk distributions, appointment states, and top conditions.

---

## Directory Structure

```text
medical_backend/
├── .env                         # Environment configurations (URI, JWT secrets, ports)
├── config.py                    # App configuration loader
├── database.py                  # PyMongo client & collection manager with indexing
├── security.py                  # Password hashing (bcrypt) & JWT token handling
├── schemas.py                   # Pydantic v2 validation models
├── ml_engine.py                 # Disease inference engine & fuzzy matching
├── main.py                      # FastAPI app entrypoint with all API endpoints
├── test_backend.py              # Automated pytest test suite
├── medassist_disease_model.pkl  # Trained ML disease model
├── model_features.pkl           # Feature columns alignment artifact
└── requirements.txt             # Project dependencies
```

---

## Setup & Execution

### 1. Requirements Installation

Ensure Python 3.12 is active and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run the Development Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive documentation will be accessible at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## API Reference

### Health
- `GET /api/health`: Check MongoDB connection, ML model status, and service uptime.

### Authentication & Profiles
- `POST /api/auth/signup`: Register a `patient` or `doctor` account.
- `POST /api/auth/login`: Authenticate and obtain a 24-hour JWT token.

### AI Diagnostics & Clinical Engine (Protected)
- `POST /api/predict`: Evaluate symptoms, run ML inference, classify risk level, and record to `symptoms_log`.
- `POST /api/recommendations`: Return clinical treatment protocols, lifestyle advice, precautions, and warning signs.

### Doctor Directory & Patient Management (Protected)
- `GET /api/doctors`: Fetch registered doctors with their specializations and locations.
- `GET /api/patients`: Fetch patient diagnostic logs combined with patient profile metadata for physician dashboards.

### Appointments (Protected)
- `POST /api/appointments`: Book a new consultation appointment (`status: 'Pending'`).
- `GET /api/appointments/{email}?role=patient|doctor`: Retrieve appointments filtered by user email and role.
- `PUT /api/appointments/{appt_id}`: Accept/reject appointments and assign scheduled time slots.

### Analytics (Protected)
- `GET /api/analytics`: Aggregate diagnostics count, risk distribution, top diseases, and appointment stats.

---

## Running the Automated Test Suite

Execute the comprehensive test suite with pytest:
```bash
pytest test_backend.py -v
```
