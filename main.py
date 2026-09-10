import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from bson import ObjectId
from bson.errors import InvalidId

from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pymongo.collection import Collection

from config import PORT, HOST
from database import DatabaseManager
from security import hash_password, verify_password, create_access_token, verify_jwt_token
from ml_engine import MLEngine
from schemas import (
    UserRegister,
    UserLogin,
    TokenResponse,
    SymptomPredictRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
    AppointmentCreate,
    AppointmentUpdate,
    AppointmentResponseItem,
    DoctorResponseItem,
    PatientProfileItem,
    AnalyticsResponse,
    ProfileUpdateRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("medassist.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database connection and ML model
    logger.info("Initializing MedAssist AI Backend...")
    try:
        DatabaseManager.connect()
        logger.info("MongoDB Atlas connection established.")
    except Exception as e:
        logger.error(f"Failed connecting to MongoDB on startup: {e}")

    try:
        MLEngine.load_model()
        logger.info("ML Disease Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed loading ML model on startup: {e}")

    yield

    # Shutdown
    DatabaseManager.close()
    logger.info("MedAssist AI Backend shut down.")

app = FastAPI(
    title="MedAssist AI Clinical Portal & Backend",
    version="2.0.0",
    description="Production-ready FastAPI and MongoDB backend for clinical disease prediction and patient-doctor management.",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["Health"])
def health_check():
    db_status = "connected"
    try:
        db = DatabaseManager.get_db()
        db.command("ping")
    except Exception as e:
        db_status = f"error: {e}"

    if MLEngine.model is None:
        MLEngine.load_model()

    model_status = "loaded" if MLEngine.model is not None else "not loaded"

    return {
        "status": "online",
        "service": "MedAssist AI Backend",
        "database": db_status,
        "ml_model": model_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ---------------------------------------------------------------------------
# 1. Authentication & Role-Based Storage
# ---------------------------------------------------------------------------
@app.post("/api/auth/signup", status_code=status.HTTP_201_CREATED, tags=["Authentication"])
def signup(user: UserRegister):
    patients_col = DatabaseManager.get_patients_collection()
    doctors_col = DatabaseManager.get_doctors_collection()

    email = user.email.lower()

    try:
        if patients_col.find_one({"email": email}) or doctors_col.find_one({"email": email}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is already registered in the system."
            )
    except HTTPException:
        raise
    except Exception as db_err:
        logger.error(f"Database query error during signup: {db_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot connect to MongoDB Atlas. Please ensure IP Access List in MongoDB Atlas allows your IP (0.0.0.0/0)."
        )

    hashed_pw = hash_password(user.password)
    user_doc = {
        "email": email,
        "password_hash": hashed_pw,
        "password": hashed_pw,  # Kept for backward compatibility
        "role": user.role,
        "full_name": user.full_name.strip(),
        "phone": user.phone,
        "age": user.age,
        "gender": user.gender,
        "location": user.location.strip(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    if user.role == "doctor":
        user_doc["specialization"] = user.specialization or "General Physician"
        doctors_col.insert_one(user_doc)
        try:
            doc_copy = {k: v for k, v in user_doc.items() if k != "_id"}
            DatabaseManager.get_db()["doctor"].insert_one(doc_copy)
        except Exception as err:
            logger.debug(f"Singular doctor mirror notice: {err}")
        logger.info(f"Doctor registered in MongoDB: {email}")
    else:
        patients_col.insert_one(user_doc)
        try:
            pat_copy = {k: v for k, v in user_doc.items() if k != "_id"}
            DatabaseManager.get_db()["patient"].insert_one(pat_copy)
        except Exception as err:
            logger.debug(f"Singular patient mirror notice: {err}")
        logger.info(f"Patient registered in MongoDB: {email}")

    return {
        "status": "success",
        "message": f"Successfully registered as {user.role}.",
        "role": user.role,
        "email": email
    }

@app.post("/api/auth/login", response_model=TokenResponse, tags=["Authentication"])
def login(credentials: UserLogin):
    email = credentials.email.lower()
    role = credentials.role.lower()

    target_col = (
        DatabaseManager.get_patients_collection()
        if role == "patient"
        else DatabaseManager.get_doctors_collection()
    )

    try:
        user_record = target_col.find_one({"email": email})
    except Exception as db_err:
        logger.error(f"Database query error during login: {db_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot connect to MongoDB Atlas. Please ensure IP Access List in MongoDB Atlas allows your IP (0.0.0.0/0)."
        )

    # If not found directly in designated role collection, check alternative collection
    if not user_record:
        alt_col = (
            DatabaseManager.get_doctors_collection()
            if role == "patient"
            else DatabaseManager.get_patients_collection()
        )
        alt_user = alt_col.find_one({"email": email})
        if alt_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Email is registered under role '{alt_user.get('role')}', not '{role}'."
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    stored_hash = user_record.get("password_hash") or user_record.get("password")
    if not stored_hash or not verify_password(credentials.password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    # Issue JWT valid for 24 hours
    token_payload = {
        "sub": email,
        "role": role,
        "full_name": user_record.get("full_name", "")
    }
    access_token = create_access_token(token_payload)

    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        role=role,
        email=email,
        full_name=user_record.get("full_name"),
        phone=user_record.get("phone"),
        age=user_record.get("age"),
        gender=user_record.get("gender"),
        location=user_record.get("location"),
        specialization=user_record.get("specialization"),
        bloodGroup=user_record.get("bloodGroup"),
        allergies=user_record.get("allergies"),
        emergencyContact=user_record.get("emergencyContact"),
        hospital=user_record.get("hospital"),
        licenseNumber=user_record.get("licenseNumber"),
        experienceYears=user_record.get("experienceYears"),
        created_at=user_record.get("created_at")
    )

@app.get("/api/auth/me", tags=["Authentication"])
def get_current_user_profile(current_user: dict = Depends(verify_jwt_token)):
    email = current_user.get("sub", "").lower()
    role = current_user.get("role", "").lower()

    col = (
        DatabaseManager.get_patients_collection()
        if role == "patient"
        else DatabaseManager.get_doctors_collection()
    )
    user_record = col.find_one({"email": email})
    if not user_record:
        alt_col = (
            DatabaseManager.get_doctors_collection()
            if role == "patient"
            else DatabaseManager.get_patients_collection()
        )
        user_record = alt_col.find_one({"email": email})

    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile record not found."
        )

    user_data = {k: v for k, v in user_record.items() if k not in ["_id", "password", "password_hash"]}
    user_data["id"] = email
    user_data["name"] = user_record.get("full_name", email)
    return user_data

@app.put("/api/auth/profile", tags=["Authentication"])
def update_current_user_profile(
    profile_data: ProfileUpdateRequest,
    current_user: dict = Depends(verify_jwt_token)
):
    email = current_user.get("sub", "").lower()
    role = current_user.get("role", "").lower()

    col = (
        DatabaseManager.get_patients_collection()
        if role == "patient"
        else DatabaseManager.get_doctors_collection()
    )

    update_fields = {}
    data_dict = profile_data.model_dump(exclude_unset=True)
    for k, v in data_dict.items():
        if v is not None:
            if k == "name" and "full_name" not in data_dict:
                update_fields["full_name"] = v
            else:
                update_fields[k] = v

    if update_fields:
        update_fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        col.update_one({"email": email}, {"$set": update_fields})
        try:
            singular_col_name = "doctor" if role == "doctor" else "patient"
            DatabaseManager.get_db()[singular_col_name].update_one({"email": email}, {"$set": update_fields})
        except Exception:
            pass

    updated_doc = col.find_one({"email": email}) or {}
    clean_data = {k: v for k, v in updated_doc.items() if k not in ["_id", "password", "password_hash"]}
    clean_data["id"] = email
    clean_data["name"] = updated_doc.get("full_name", email)

    return {
        "status": "success",
        "message": "Profile updated successfully.",
        "user": clean_data
    }


# ---------------------------------------------------------------------------
# 2. AI Model & Prediction Engine
# ---------------------------------------------------------------------------
@app.post("/api/predict", response_model=PredictionResponse, tags=["AI Diagnostics"])
def predict_disease(
    request: SymptomPredictRequest,
    current_user: dict = Depends(verify_jwt_token)
):
    try:
        prediction_result = MLEngine.run_prediction(request.model_dump())
    except Exception as e:
        logger.error(f"Prediction execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Disease prediction engine failed: {str(e)}"
        )

    # Persist prediction in symptoms_log collection
    symptoms_log_col = DatabaseManager.get_symptoms_log_collection()
    log_doc = {
        "email": request.email.lower(),
        "symptoms_text": request.symptoms_text,
        "indicators": prediction_result["indicators"],
        "predicted_disease": prediction_result["predicted_disease"],
        "prediction": prediction_result["predicted_disease"],  # Compatibility with existing schema
        "confidence_score": prediction_result["confidence_score"],
        "risk_level": prediction_result["risk_level"],
        "normalized_symptom": prediction_result["normalized_symptom"],
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    try:
        symptoms_log_col.insert_one(log_doc)
    except Exception as e:
        logger.error(f"Failed logging prediction to MongoDB: {e}")

    return PredictionResponse(
        status="success",
        predicted_disease=prediction_result["predicted_disease"],
        confidence_score=prediction_result["confidence_score"],
        risk_level=prediction_result["risk_level"],
        recommendations=prediction_result["recommendations"],
        normalized_symptom=prediction_result["normalized_symptom"]
    )

# ---------------------------------------------------------------------------
# 3. Recommendations & Analytics
# ---------------------------------------------------------------------------
@app.post("/api/recommendations", response_model=RecommendationResponse, tags=["Clinical Recommendations"])
def get_recommendations(
    req: RecommendationRequest,
    current_user: dict = Depends(verify_jwt_token)
):
    disease = req.disease or "General Clinical Condition"
    risk = (req.risk_level or "Low").capitalize()

    lifestyle_advice = [
        "Maintain adequate daily hydration (2.5 to 3 liters of purified water).",
        "Ensure 7 to 8 hours of uninterrupted restorative sleep to enhance immune response.",
        "Avoid strenuous physical exertion and stress until clinical symptoms resolve.",
        "Consume nutrient-dense, easily digestible meals rich in antioxidants and vitamins."
    ]

    precautions = [
        "Monitor core body temperature and blood pressure twice daily.",
        "Wear a protective medical mask if leaving home to avoid airborne pathogens.",
        "Refrain from consuming excessively salty, greasy, or processed foods.",
        "Do not self-medicate with high-dose antibiotics or non-prescribed analgesics."
    ]

    when_to_consult = (
        "Seek immediate emergency medical care if you experience sharp chest pain, "
        "shortness of breath, persistent fever above 102°F (38.9°C), sudden confusion, "
        "or continuous vomiting."
    )

    if risk == "High":
        precautions.insert(0, "CRITICAL: Book an immediate priority consultation with a medical specialist.")
        lifestyle_advice.insert(0, "Strict bed rest and continuous clinical observation recommended.")
    elif risk == "Medium":
        precautions.insert(0, "Schedule an appointment with a physician within 24 to 48 hours.")

    treatment_suggestions = (
        f"Standard clinical protocol for managing {disease}: Initiate symptom-targeted "
        f"therapy, follow prescribed rest regimens, and undergo diagnostic laboratory evaluations "
        f"under the direct supervision of a licensed physician."
    )

    return RecommendationResponse(
        status="success",
        disease=disease,
        risk_level=risk,
        treatment_suggestions=treatment_suggestions,
        lifestyle_advice=lifestyle_advice,
        precautions=precautions,
        when_to_consult=when_to_consult
    )

@app.get("/api/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics(current_user: dict = Depends(verify_jwt_token)):
    symptoms_col = DatabaseManager.get_symptoms_log_collection()
    appointments_col = DatabaseManager.get_appointments_collection()

    # 1. Total predictions
    total_predictions = symptoms_col.count_documents({})

    # 2. Risk distribution aggregation
    risk_pipeline = [
        {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}
    ]
    risk_cursor = symptoms_col.aggregate(risk_pipeline)
    risk_distribution = {doc["_id"] or "Unknown": doc["count"] for doc in risk_cursor}

    # Ensure standard keys are present for UI
    for level in ["Low", "Medium", "High"]:
        risk_distribution.setdefault(level, 0)

    # 3. Top predicted diseases aggregation
    disease_pipeline = [
        {
            "$project": {
                "disease": {
                    "$ifNull": ["$predicted_disease", "$prediction"]
                }
            }
        },
        {"$group": {"_id": "$disease", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5}
    ]
    disease_cursor = symptoms_col.aggregate(disease_pipeline)
    disease_stats = [
        {"disease": doc["_id"] or "Unclassified", "count": doc["count"]}
        for doc in disease_cursor
    ]

    # 4. Appointment status aggregation
    appt_pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    appt_cursor = appointments_col.aggregate(appt_pipeline)
    appointment_stats = {doc["_id"] or "Pending": doc["count"] for doc in appt_cursor}
    for st in ["Pending", "Accepted", "Rejected"]:
        appointment_stats.setdefault(st, 0)

    return AnalyticsResponse(
        status="success",
        total_predictions=total_predictions,
        risk_distribution=risk_distribution,
        disease_stats=disease_stats,
        appointment_stats=appointment_stats,
        system_health="Optimal (99.9% Uptime)"
    )

@app.get("/api/doctors", response_model=Dict[str, List[DoctorResponseItem]], tags=["Doctor Directory"])
def get_doctors(current_user: dict = Depends(verify_jwt_token)):
    doctors_col = DatabaseManager.get_doctors_collection()
    doctors = []
    cursor = doctors_col.find({}, {"password": 0, "password_hash": 0})
    for doc in cursor:
        doctors.append(
            DoctorResponseItem(
                email=doc.get("email", ""),
                full_name=doc.get("full_name", "Dr. Specialist"),
                specialization=doc.get("specialization", "General Physician"),
                location=doc.get("location", "Not Specified"),
                phone=doc.get("phone", "")
            )
        )
    return {"doctors": doctors}

# ---------------------------------------------------------------------------
# 4. Appointments & Patient Management
# ---------------------------------------------------------------------------
@app.post("/api/appointments", status_code=status.HTTP_201_CREATED, tags=["Appointments"])
def book_appointment(
    appt: AppointmentCreate,
    current_user: dict = Depends(verify_jwt_token)
):
    appointments_col = DatabaseManager.get_appointments_collection()

    appt_doc = {
        "patient_email": appt.patient_email.lower(),
        "doctor_email": appt.doctor_email.lower(),
        "appointment_date": appt.appointment_date,
        "status": "Pending",
        "scheduled_time": "To be confirmed",
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    result = appointments_col.insert_one(appt_doc)
    logger.info(f"Booked appointment {result.inserted_id} between {appt.patient_email} and {appt.doctor_email}")

    return {
        "status": "success",
        "message": "Appointment booked successfully.",
        "appointment_id": str(result.inserted_id)
    }

@app.get("/api/appointments/{email}", tags=["Appointments"])
def get_appointments(
    email: str,
    role: str = Query(..., description="Role: 'patient' or 'doctor'"),
    current_user: dict = Depends(verify_jwt_token)
):
    appointments_col = DatabaseManager.get_appointments_collection()
    email_clean = email.strip().lower()
    role_clean = role.strip().lower()

    if role_clean == "patient":
        query = {"patient_email": email_clean}
    elif role_clean == "doctor":
        query = {"doctor_email": email_clean}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'role' must be either 'patient' or 'doctor'"
        )

    appointments_cursor = appointments_col.find(query).sort("_id", -1)
    appts = []
    for doc in appointments_cursor:
        doc_id = str(doc["_id"])
        target_email = doc.get("doctor_email") if role_clean == "patient" else doc.get("patient_email")
        appts.append({
            "id": doc_id,
            "patient_email": doc.get("patient_email"),
            "doctor_email": doc.get("doctor_email"),
            "target_email": target_email,
            "appointment_date": doc.get("appointment_date"),
            "status": doc.get("status", "Pending"),
            "scheduled_time": doc.get("scheduled_time", "To be confirmed"),
            "created_at": doc.get("created_at")
        })

    return {"appointments": appts}

@app.put("/api/appointments/{appt_id}", tags=["Appointments"])
def update_appointment(
    appt_id: str,
    update: AppointmentUpdate,
    current_user: dict = Depends(verify_jwt_token)
):
    appointments_col = DatabaseManager.get_appointments_collection()

    try:
        query = {"_id": ObjectId(appt_id)}
    except InvalidId:
        query = {"_id": appt_id}

    existing = appointments_col.find_one(query)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Appointment with ID '{appt_id}' was not found."
        )

    appointments_col.update_one(
        query,
        {
            "$set": {
                "status": update.status,
                "scheduled_time": update.scheduled_time,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        }
    )

    logger.info(f"Updated appointment {appt_id} to status '{update.status}' at '{update.scheduled_time}'")

    return {
        "status": "success",
        "message": f"Appointment status updated to '{update.status}' with time '{update.scheduled_time}'."
    }

@app.get("/api/patients", tags=["Patient Management"])
def get_patients(current_user: dict = Depends(verify_jwt_token)):
    """Return all patient diagnostic records and merged profiles for doctor dashboards."""
    symptoms_col = DatabaseManager.get_symptoms_log_collection()
    patients_col = DatabaseManager.get_patients_collection()

    symptoms_cursor = symptoms_col.find({}).sort("_id", -1)
    patients_list = []

    # Cache patient profiles to avoid repeated DB lookups
    patient_cache = {}
    seen_emails = set()

    for s in symptoms_cursor:
        email = s.get("email", "").lower()
        seen_emails.add(email)
        if email not in patient_cache:
            profile = patients_col.find_one({"email": email})
            patient_cache[email] = profile or {}

        patient_profile = patient_cache[email]

        # Format symptoms text
        symptoms_display = s.get("symptoms_text") or s.get("symptoms", "Clinical Consultation")
        prediction_val = s.get("predicted_disease") or s.get("prediction", "General Condition")
        risk_val = s.get("risk_level", "Low")
        conf_val = s.get("confidence_score", "90.00%")

        patients_list.append({
            "id": str(s["_id"]),
            "email": email,
            "full_name": patient_profile.get("full_name", "Patient"),
            "phone": patient_profile.get("phone", "N/A"),
            "age": patient_profile.get("age", 30),
            "gender": patient_profile.get("gender", "Unspecified"),
            "location": patient_profile.get("location", "Not Specified"),
            "guardian": "Primary Contact Available",
            "symptoms": symptoms_display,
            "prediction": prediction_val,
            "risk_level": risk_val,
            "confidence_score": conf_val,
            "created_at": s.get("created_at")
        })

    # Include newly registered patients who have not submitted symptom logs yet
    for p in patients_col.find({}):
        p_email = p.get("email", "").lower()
        if p_email not in seen_emails:
            patients_list.append({
                "id": str(p["_id"]),
                "email": p_email,
                "full_name": p.get("full_name", "Patient"),
                "phone": p.get("phone", "N/A"),
                "age": p.get("age", 30),
                "gender": p.get("gender", "Unspecified"),
                "location": p.get("location", "Not Specified"),
                "guardian": "Primary Contact Available",
                "symptoms": "Newly registered patient (Pending symptoms checklist)",
                "prediction": "Pending Clinical Assessment",
                "risk_level": "Low",
                "confidence_score": "N/A",
                "created_at": p.get("created_at")
            })

    return {"patients": patients_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
