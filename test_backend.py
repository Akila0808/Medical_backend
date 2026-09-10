import pytest
import uuid
from fastapi.testclient import TestClient
from main import app
from database import DatabaseManager

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    db = DatabaseManager.connect()
    yield
    # Clean up test accounts so user database stays clean
    try:
        db["patients"].delete_many({"email": {"$regex": "@medassist\\.com$"}})
        db["doctors"].delete_many({"email": {"$regex": "@medassist\\.com$"}})
        db["patient"].delete_many({"email": {"$regex": "@medassist\\.com$"}})
        db["doctor"].delete_many({"email": {"$regex": "@medassist\\.com$"}})
        db["symptoms_log"].delete_many({"email": {"$regex": "@medassist\\.com$"}})
        db["appointments"].delete_many({"patient_email": {"$regex": "@medassist\\.com$"}})
    except Exception as e:
        print("Teardown notice:", e)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["database"] == "connected"
    assert data["ml_model"] == "loaded"

def test_signup_validation_failures():
    # 1. Weak password (no uppercase, no special char)
    resp = client.post("/api/auth/signup", json={
        "email": f"test_{uuid.uuid4().hex[:6]}@example.com",
        "password": "simplepassword1",
        "role": "patient",
        "full_name": "Test User",
        "phone": "9876543210",
        "age": 25,
        "gender": "Male",
        "location": "Chennai"
    })
    assert resp.status_code == 422 or resp.status_code == 400

    # 2. Invalid role
    resp = client.post("/api/auth/signup", json={
        "email": f"test_{uuid.uuid4().hex[:6]}@example.com",
        "password": "Password@123",
        "role": "admin",
        "full_name": "Test User",
        "phone": "9876543210",
        "age": 25,
        "gender": "Male",
        "location": "Chennai"
    })
    assert resp.status_code == 422 or resp.status_code == 400

    # 3. Invalid phone number (letters / wrong length)
    resp = client.post("/api/auth/signup", json={
        "email": f"test_{uuid.uuid4().hex[:6]}@example.com",
        "password": "Password@123",
        "role": "patient",
        "full_name": "Test User",
        "phone": "12345",
        "age": 25,
        "gender": "Male",
        "location": "Chennai"
    })
    assert resp.status_code == 422 or resp.status_code == 400

def test_auth_full_flow():
    unique_suffix = uuid.uuid4().hex[:8]
    patient_email = f"patient_{unique_suffix}@medassist.com"
    doctor_email = f"doctor_{unique_suffix}@medassist.com"
    password = "SecurePassword@2026"

    # 1. Register Patient
    patient_payload = {
        "email": patient_email,
        "password": password,
        "role": "patient",
        "full_name": "Ravi Kumar",
        "phone": "9876543210",
        "age": 32,
        "gender": "Male",
        "location": "Chennai"
    }
    resp = client.post("/api/auth/signup", json=patient_payload)
    assert resp.status_code == 201
    assert resp.json()["role"] == "patient"

    # Verify duplicate email prevention
    resp_dup = client.post("/api/auth/signup", json=patient_payload)
    assert resp_dup.status_code == 400
    assert "already registered" in resp_dup.json()["detail"].lower()

    # 2. Register Doctor
    doctor_payload = {
        "email": doctor_email,
        "password": password,
        "role": "doctor",
        "full_name": "Dr. Ananya Sharma",
        "phone": "9123456780",
        "age": 42,
        "gender": "Female",
        "location": "Bengaluru",
        "specialization": "Cardiologist"
    }
    resp = client.post("/api/auth/signup", json=doctor_payload)
    assert resp.status_code == 201
    assert resp.json()["role"] == "doctor"

    # 3. Login Patient
    resp = client.post("/api/auth/login", json={
        "email": patient_email,
        "password": password,
        "role": "patient"
    })
    assert resp.status_code == 200
    token_data = resp.json()
    assert "access_token" in token_data
    patient_token = token_data["access_token"]
    assert token_data["role"] == "patient"
    assert token_data["full_name"] == "Ravi Kumar"
    assert token_data["phone"] == "9876543210"
    assert token_data["age"] == 32
    assert token_data["gender"] == "Male"
    assert token_data["location"] == "Chennai"

    # 4. Login Doctor
    resp = client.post("/api/auth/login", json={
        "email": doctor_email,
        "password": password,
        "role": "doctor"
    })
    assert resp.status_code == 200
    doc_token_data = resp.json()
    doctor_token = doc_token_data["access_token"]
    assert doc_token_data["specialization"] == "Cardiologist"
    assert doc_token_data["phone"] == "9123456780"
    assert doc_token_data["location"] == "Bengaluru"

    # 4b. Test GET /api/auth/me for patient & doctor
    resp_me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {patient_token}"})
    assert resp_me.status_code == 200
    me_data = resp_me.json()
    assert me_data["email"] == patient_email
    assert me_data["phone"] == "9876543210"
    assert me_data["age"] == 32

    # 4c. Test PUT /api/auth/profile
    resp_update = client.put(
        "/api/auth/profile",
        json={"location": "Kochi, Kerala", "allergies": "Pollen"},
        headers={"Authorization": f"Bearer {patient_token}"}
    )
    assert resp_update.status_code == 200
    assert resp_update.json()["user"]["location"] == "Kochi, Kerala"
    assert resp_update.json()["user"]["allergies"] == "Pollen"

    # 5. Invalid password login
    resp = client.post("/api/auth/login", json={
        "email": patient_email,
        "password": "WrongPassword@123",
        "role": "patient"
    })
    assert resp.status_code == 401

    # 6. Wrong role login
    resp = client.post("/api/auth/login", json={
        "email": patient_email,
        "password": password,
        "role": "doctor"
    })
    assert resp.status_code == 401

    headers = {"Authorization": f"Bearer {patient_token}"}
    doc_headers = {"Authorization": f"Bearer {doctor_token}"}

    # 7. Disease Prediction (Protected)
    predict_payload = {
        "email": patient_email,
        "symptoms_text": "Severe fever with chills, body fatigue and joint pain",
        "fever": "Yes",
        "cough": "No",
        "fatigue": "Yes",
        "difficulty_breathing": "No",
        "blood_pressure": "High",
        "cholesterol": "Normal"
    }
    resp = client.post("/api/predict", json=predict_payload, headers=headers)
    assert resp.status_code == 200
    pred_data = resp.json()
    assert "predicted_disease" in pred_data
    assert "confidence_score" in pred_data
    assert pred_data["risk_level"] in ["Low", "Medium", "High"]
    assert pred_data["status"] == "success"

    # Verify prediction persisted in MongoDB symptoms_log
    symptoms_col = DatabaseManager.get_symptoms_log_collection()
    logged = symptoms_col.find_one({"email": patient_email})
    assert logged is not None
    assert logged["predicted_disease"] == pred_data["predicted_disease"]

    # 8. Clinical Recommendations (Protected)
    rec_payload = {
        "disease": pred_data["predicted_disease"],
        "risk_level": pred_data["risk_level"]
    }
    resp = client.post("/api/recommendations", json=rec_payload, headers=headers)
    assert resp.status_code == 200
    rec_data = resp.json()
    assert rec_data["status"] == "success"
    assert len(rec_data["lifestyle_advice"]) > 0
    assert len(rec_data["precautions"]) > 0
    assert "when_to_consult" in rec_data

    # 9. Fetch Doctors Directory (Protected)
    resp = client.get("/api/doctors", headers=headers)
    assert resp.status_code == 200
    doc_list = resp.json()["doctors"]
    assert any(d["email"] == doctor_email for d in doc_list)
    # Ensure passwords are not leaked
    for d in doc_list:
        assert "password" not in d
        assert "password_hash" not in d

    # 10. Book Appointment (Protected)
    appt_payload = {
        "patient_email": patient_email,
        "doctor_email": doctor_email,
        "appointment_date": "2026-10-15"
    }
    resp = client.post("/api/appointments", json=appt_payload, headers=headers)
    assert resp.status_code == 201
    appt_id = resp.json()["appointment_id"]
    assert appt_id is not None

    # 11. Fetch Appointments for Patient
    resp = client.get(f"/api/appointments/{patient_email}?role=patient", headers=headers)
    assert resp.status_code == 200
    patient_appts = resp.json()["appointments"]
    assert len(patient_appts) >= 1
    assert patient_appts[0]["status"] == "Pending"

    # 12. Fetch Appointments for Doctor
    resp = client.get(f"/api/appointments/{doctor_email}?role=doctor", headers=doc_headers)
    assert resp.status_code == 200
    doctor_appts = resp.json()["appointments"]
    assert len(doctor_appts) >= 1
    assert doctor_appts[0]["id"] == appt_id

    # 13. Update Appointment by Doctor
    update_payload = {
        "status": "Accepted",
        "scheduled_time": "10:30 AM"
    }
    resp = client.put(f"/api/appointments/{appt_id}", json=update_payload, headers=doc_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # Verify updated appointment
    resp = client.get(f"/api/appointments/{patient_email}?role=patient", headers=headers)
    updated_appt = [a for a in resp.json()["appointments"] if a["id"] == appt_id][0]
    assert updated_appt["status"] == "Accepted"
    assert updated_appt["scheduled_time"] == "10:30 AM"

    # 14. Doctor Dashboard: Patient profiles & diagnostic records
    resp = client.get("/api/patients", headers=doc_headers)
    assert resp.status_code == 200
    patients_data = resp.json()["patients"]
    assert len(patients_data) >= 1
    # Check that patient diagnostic record is joined with patient profile
    patient_record = [p for p in patients_data if p["email"] == patient_email][0]
    assert patient_record["full_name"] == "Ravi Kumar"
    assert patient_record["phone"] == "9876543210"
    assert patient_record["age"] == 32
    assert patient_record["gender"] == "Male"
    assert patient_record["location"] == "Kochi, Kerala"

    # 15. Analytics (Protected)
    resp = client.get("/api/analytics", headers=doc_headers)
    assert resp.status_code == 200
    analytics = resp.json()
    assert analytics["total_predictions"] >= 1
    assert "Low" in analytics["risk_distribution"]
    assert "Pending" in analytics["appointment_stats"] or "Accepted" in analytics["appointment_stats"]
    assert analytics["system_health"] == "Optimal (99.9% Uptime)"

def test_unauthorized_endpoints():
    # Protected endpoints must reject unauthenticated requests
    resp = client.post("/api/predict", json={})
    assert resp.status_code == 401

    resp = client.post("/api/recommendations", json={})
    assert resp.status_code == 401

    resp = client.get("/api/doctors")
    assert resp.status_code == 401

    resp = client.post("/api/appointments", json={})
    assert resp.status_code == 401

    resp = client.get("/api/appointments/test@example.com?role=patient")
    assert resp.status_code == 401

    resp = client.put("/api/appointments/nonexistent_id", json={"status": "Accepted", "scheduled_time": "10:00 AM"})
    assert resp.status_code == 401

    resp = client.get("/api/patients")
    assert resp.status_code == 401

    resp = client.get("/api/analytics")
    assert resp.status_code == 401

def test_appointment_edge_cases():
    from security import create_access_token
    token = create_access_token({"sub": "admin@example.com", "role": "doctor"})
    headers = {"Authorization": f"Bearer {token}"}

    # Invalid role in appointments query
    resp = client.get("/api/appointments/test@example.com?role=invalid_role", headers=headers)
    assert resp.status_code == 400

    # Non-existent appointment update
    resp = client.put("/api/appointments/507f1f77bcf86cd799439011", json={"status": "Accepted", "scheduled_time": "12:00 PM"}, headers=headers)
    assert resp.status_code == 404

def test_ml_clinical_safety_overrides():
    from ml_engine import MLEngine
    # 1. Critical chest pain presentation
    pred1 = MLEngine.run_prediction({
        "symptoms_text": "Sudden severe chest pain radiating to left arm",
        "fever": "No",
        "cough": "No",
        "fatigue": "Yes",
        "difficulty_breathing": "Yes",
        "blood_pressure": "High",
        "cholesterol": "High"
    })
    assert pred1["risk_level"] == "High"
    assert "Cardiac" in pred1["predicted_disease"] or "Ischemia" in pred1["predicted_disease"]

    # 2. GI bleeding presentation
    pred2 = MLEngine.run_prediction({
        "symptoms_text": "Patient has blood in vomit and acute abdominal distress",
        "fever": "No",
        "cough": "No",
        "fatigue": "Yes",
        "difficulty_breathing": "No",
        "blood_pressure": "Normal",
        "cholesterol": "Normal"
    })
    assert pred2["risk_level"] == "High"
    assert "Bleeding" in pred2["predicted_disease"]

def test_existing_collections_preserved():
    # Verify that existing collections are intact and not modified
    db = DatabaseManager.get_db()
    assert "Sample_diseases" in db.list_collection_names()
    assert "Severity" in db.list_collection_names()
    assert "categories" in db.list_collection_names()

    # Document counts should be positive
    assert db["Sample_diseases"].count_documents({}) == 300
    assert db["Severity"].count_documents({}) == 133
    assert db["categories"].count_documents({}) == 5000

if __name__ == "__main__":
    pytest.main(["-v", "test_backend.py"])

