import re
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field, field_validator

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password with minimum 8 characters")
    role: str = Field(..., description="Role: patient or doctor")
    full_name: str = Field(..., min_length=2, max_length=100)
    phone: str = Field(..., description="10-digit phone number")
    age: int = Field(..., ge=1, le=120)
    gender: str = Field(..., min_length=1)
    location: str = Field(..., min_length=2)
    specialization: Optional[str] = "General Physician"

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        role_lower = v.strip().lower()
        if role_lower not in ["patient", "doctor"]:
            raise ValueError("Role must be either 'patient' or 'doctor'")
        return role_lower

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must include at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must include at least one lowercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must include at least one numeric digit")
        if not re.search(r"[\W_]", v):
            raise ValueError("Password must include at least one special character")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        cleaned = re.sub(r"[\s\-\(\)\+]", "", v)
        # Check if contains standard 10-digit or 10-12 digits with country code
        if not re.fullmatch(r"\d{10,12}", cleaned):
            raise ValueError("Phone number must be a valid 10-digit number")
        # Keep last 10 digits as standardized 10-digit phone
        return cleaned[-10:]

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    role: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        role_lower = v.strip().lower()
        if role_lower not in ["patient", "doctor"]:
            raise ValueError("Role must be either 'patient' or 'doctor'")
        return role_lower

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    role: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    specialization: Optional[str] = None
    bloodGroup: Optional[str] = None
    allergies: Optional[str] = None
    emergencyContact: Optional[str] = None
    hospital: Optional[str] = None
    licenseNumber: Optional[str] = None
    experienceYears: Optional[int] = None
    created_at: Optional[str] = None

class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    specialization: Optional[str] = None
    bloodGroup: Optional[str] = None
    allergies: Optional[str] = None
    emergencyContact: Optional[str] = None
    hospital: Optional[str] = None
    licenseNumber: Optional[str] = None
    experienceYears: Optional[int] = None

class SymptomPredictRequest(BaseModel):
    email: EmailStr
    symptoms_text: str = Field(..., min_length=1)
    fever: str = Field(default="No")
    cough: str = Field(default="No")
    fatigue: str = Field(default="No")
    difficulty_breathing: str = Field(default="No")
    blood_pressure: str = Field(default="Normal")
    cholesterol: str = Field(default="Normal")

    @field_validator("fever", "cough", "fatigue", "difficulty_breathing")
    @classmethod
    def normalize_yes_no(cls, v: str) -> str:
        val = v.strip().capitalize()
        return val if val in ["Yes", "No"] else "No"

    @field_validator("blood_pressure", "cholesterol")
    @classmethod
    def normalize_levels(cls, v: str) -> str:
        val = v.strip().capitalize()
        return val if val in ["Normal", "High", "Low"] else "Normal"

class PredictionResponse(BaseModel):
    status: str = "success"
    predicted_disease: str
    confidence_score: str
    risk_level: str
    recommendations: str
    normalized_symptom: Optional[str] = None

class RecommendationRequest(BaseModel):
    disease: Optional[str] = "General Condition"
    risk_level: Optional[str] = "Low"

class RecommendationResponse(BaseModel):
    status: str = "success"
    disease: str
    risk_level: str
    treatment_suggestions: str
    lifestyle_advice: List[str]
    precautions: List[str]
    when_to_consult: str

class AppointmentCreate(BaseModel):
    patient_email: EmailStr
    doctor_email: EmailStr
    appointment_date: str = Field(..., min_length=4)

class AppointmentUpdate(BaseModel):
    status: str = Field(..., description="'Accepted' or 'Rejected'")
    scheduled_time: str = Field(..., min_length=1)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        val = v.strip().capitalize()
        if val not in ["Accepted", "Rejected", "Pending"]:
            raise ValueError("Status must be 'Accepted', 'Rejected', or 'Pending'")
        return val

class AppointmentResponseItem(BaseModel):
    id: str
    patient_email: str
    doctor_email: str
    target_email: Optional[str] = None
    appointment_date: str
    status: str
    scheduled_time: str
    created_at: Optional[str] = None

class DoctorResponseItem(BaseModel):
    email: str
    full_name: str
    specialization: str
    location: str
    phone: Optional[str] = None

class PatientProfileItem(BaseModel):
    id: str
    email: str
    full_name: str
    phone: str
    age: int
    gender: str
    location: str
    symptoms: str
    prediction: str
    risk_level: str
    confidence_score: Optional[str] = None
    created_at: Optional[str] = None

class AnalyticsResponse(BaseModel):
    status: str = "success"
    total_predictions: int
    risk_distribution: Dict[str, int]
    disease_stats: List[Dict[str, Any]]
    appointment_stats: Dict[str, int]
    system_health: str = "Optimal (99.9% Uptime)"
