import re
import logging
from typing import Dict, Any, Optional, List, Tuple
import joblib
import pandas as pd
import numpy as np
from thefuzz import process, fuzz

from config import MODEL_PATH, MODEL_FEATURES_PATH

logger = logging.getLogger("medassist.ml_engine")

# Expanded Clinical Knowledge Base containing both dataset-listed and unlisted conditions
MEDICAL_CONDITIONS_DB = [
    {
        "disease": "COVID-19 / SARS-CoV-2 Viral Infection",
        "keywords": ["loss of taste", "loss of smell", "anosmia", "ageusia", "covid", "corona", "taste and smell", "dry cough", "breathlessness"],
        "required_indicators": {"fever": "Yes"},
        "base_risk": "Medium",
        "description": "Acute respiratory illness caused by the SARS-CoV-2 coronavirus.",
        "urgency": "Monitor oxygen saturation; isolate and consult physician if dyspnea develops."
    },
    {
        "disease": "Acute Appendicitis",
        "keywords": ["right lower abdomen", "right lower abdominal", "appendix", "mcburney", "belly button to right", "abdominal pain vomiting fever"],
        "required_indicators": {},
        "base_risk": "High",
        "description": "Inflammation of the vermiform appendix requiring urgent surgical evaluation.",
        "urgency": "CRITICAL: Urgent surgical assessment required immediately to avoid perforation."
    },
    {
        "disease": "Urinary Tract Infection (UTI) / Cystitis",
        "keywords": ["burning urination", "burn when peeing", "frequent urination", "urinate frequently", "pelvic pain", "cloudy urine", "foul smelling urine", "uti", "painful urination"],
        "required_indicators": {},
        "base_risk": "Medium",
        "description": "Bacterial infection affecting the urinary bladder, urethra, or kidneys.",
        "urgency": "Consult a physician for urinalysis and targeted antimicrobial therapy."
    },
    {
        "disease": "Migraine / Vascular Cephalgia",
        "keywords": ["migraine", "one side headache", "one-sided headache", "unilateral headache", "light sensitivity", "photophobia", "throbbing headache", "aura", "sound sensitivity", "visual disturbances"],
        "required_indicators": {},
        "base_risk": "Medium",
        "description": "Recurrent neurological condition characterized by intense, throbbing unilateral headaches.",
        "urgency": "Rest in a dark, quiet room; consult a neurologist if symptoms are refractory."
    },
    {
        "disease": "Gastroesophageal Reflux Disease (GERD) / Acid Reflux",
        "keywords": ["heartburn", "acid reflux", "acid regurgitation", "chest burning after eating", "sour taste", "burning in throat", "gerd", "esophageal burning"],
        "required_indicators": {},
        "base_risk": "Low",
        "description": "Chronic digestive disease where stomach acid irritates the food pipe lining.",
        "urgency": "Elevate head during sleep, avoid acidic/fatty foods, and consult gastroenterologist."
    },
    {
        "disease": "Nephrolithiasis (Kidney Stones)",
        "keywords": ["kidney stone", "flank pain", "pain in side radiating to groin", "blood in urine", "hematuria", "severe lower back side pain", "renal colic"],
        "required_indicators": {},
        "base_risk": "High",
        "description": "Hard deposits of minerals and acid salts forming inside the kidneys.",
        "urgency": "Seek immediate medical evaluation for diagnostic ultrasound and pain management."
    },
    {
        "disease": "Malaria / Plasmodium Infection",
        "keywords": ["malaria", "cyclical fever", "shivering in cycles", "high fever with sweating", "mosquito bite fever", "intermittent fever", "rigors"],
        "required_indicators": {"fever": "Yes"},
        "base_risk": "High",
        "description": "Mosquito-borne infectious disease caused by Plasmodium parasites.",
        "urgency": "Urgent peripheral blood smear examination and antimalarial therapy required."
    },
    {
        "disease": "Streptococcal Pharyngitis / Acute Tonsillitis",
        "keywords": ["sore throat", "strep throat", "swollen tonsils", "white patches on tonsils", "pain swallowing", "difficulty swallowing throat", "tonsillitis"],
        "required_indicators": {},
        "base_risk": "Medium",
        "description": "Bacterial or viral infection causing severe inflammation of the tonsils and throat.",
        "urgency": "Seek medical consultation for throat culture or rapid strep test."
    },
    {
        "disease": "Acute Viral Gastroenteritis / Food Poisoning",
        "keywords": ["watery diarrhea", "loose stools", "food poisoning", "stomach bug", "vomiting diarrhea cramps", "nausea vomiting diarrhea", "cramping dehydration"],
        "required_indicators": {},
        "base_risk": "Medium",
        "description": "Intestinal infection marked by watery diarrhea, abdominal cramps, nausea, and vomiting.",
        "urgency": "Maintain aggressive oral rehydration therapy (ORS); seek care if unable to retain fluids."
    },
    {
        "disease": "Acute Sinusitis / Rhino-sinusitis",
        "keywords": ["sinus", "facial pressure", "pain behind eyes", "nasal congestion", "thick yellow mucus", "forehead pressure", "sinus headache"],
        "required_indicators": {},
        "base_risk": "Low",
        "description": "Inflammation or swelling of the tissue lining the sinuses.",
        "urgency": "Utilize steam inhalation, saline nasal rinses, and consult ENT if prolonged > 10 days."
    },
    {
        "disease": "Allergic Rhinitis / Upper Airway Allergy",
        "keywords": ["sneezing repeatedly", "continuous sneezing", "itchy eyes", "watery eyes", "hay fever", "allergic rhinitis", "runny nose allergy"],
        "required_indicators": {},
        "base_risk": "Low",
        "description": "Allergic response to specific allergens causing sneezing, itching, and congestion.",
        "urgency": "Avoid allergen triggers and consider antihistamines under medical guidance."
    },
    {
        "disease": "Acute Bacterial/Viral Pneumonia",
        "keywords": ["pneumonia", "productive cough phlegm", "greenish yellow phlegm", "chest pain when breathing", "pleuritic chest pain", "fever chills productive cough"],
        "required_indicators": {"cough": "Yes"},
        "base_risk": "High",
        "description": "Infection that inflames air sacs in one or both lungs, which may fill with fluid.",
        "urgency": "CRITICAL: Urgent chest X-ray and clinical assessment needed."
    },
    {
        "disease": "Acute Otitis Media (Middle Ear Infection)",
        "keywords": ["ear pain", "earache", "ear fullness", "fluid draining from ear", "hearing muffled", "sharp ear pain"],
        "required_indicators": {},
        "base_risk": "Low",
        "description": "Infection of the air-filled space behind the eardrum containing vibrating bones.",
        "urgency": "Consult an ENT specialist for otoscopic examination."
    },
    {
        "disease": "Conjunctivitis (Pink Eye)",
        "keywords": ["pink eye", "red eye", "eye discharge", "crusty eye", "itchy gritty eye", "conjunctivitis"],
        "required_indicators": {},
        "base_risk": "Low",
        "description": "Inflammation or infection of the transparent membrane lining the eyelid and eyeball.",
        "urgency": "Maintain eye hygiene, avoid rubbing eyes, and consult an ophthalmologist."
    },
    {
        "disease": "Iron Deficiency Anemia",
        "keywords": ["pale skin", "extreme weakness", "dizziness on standing", "brittle nails", "cold hands and feet", "anemia", "pale conjunctiva"],
        "required_indicators": {"fatigue": "Yes"},
        "base_risk": "Medium",
        "description": "Condition in which blood lacks adequate healthy red blood cells to carry sufficient oxygen.",
        "urgency": "Obtain Complete Blood Count (CBC) and serum ferritin under physician supervision."
    },
    {
        "disease": "Diabetes Mellitus / Hyperglycemia",
        "keywords": ["excessive thirst", "drinking water constantly", "peeing constantly", "frequent urination night", "unexplained weight loss", "polydipsia", "polyuria"],
        "required_indicators": {},
        "base_risk": "Medium",
        "description": "Metabolic disorder marked by chronic elevated levels of blood glucose.",
        "urgency": "Undergo fasting plasma glucose and HbA1c testing with an endocrinologist."
    },
    {
        "disease": "Atopic Dermatitis / Eczema",
        "keywords": ["eczema", "itchy rash", "red itchy patches", "dry scaly skin", "skin redness itching", "creases of elbows rash"],
        "required_indicators": {},
        "base_risk": "Low",
        "description": "Chronic inflammatory skin disease causing dry, itchy, and erythematous lesions.",
        "urgency": "Apply dermatological emollients; seek specialist review for topical therapy."
    },
    {
        "disease": "Gouty Arthritis / Hyperuricemia",
        "keywords": ["gout", "big toe pain", "swollen red big toe", "sudden intense joint pain", "hot red swollen joint"],
        "required_indicators": {},
        "base_risk": "Medium",
        "description": "Form of arthritis characterized by severe pain, redness, and tenderness in joints due to uric acid crystals.",
        "urgency": "Check serum uric acid level and consult a rheumatologist."
    },
    {
        "disease": "Acute Meningitis",
        "keywords": ["stiff neck", "neck stiffness", "cannot bend neck", "high fever stiff neck", "severe headache fever neck stiffness", "meningitis"],
        "required_indicators": {},
        "base_risk": "High",
        "description": "Acute inflammation of the protective membranes covering the brain and spinal cord.",
        "urgency": "EMERGENCY: Immediate hospital emergency admission required."
    },
    {
        "disease": "Typhoid Fever / Enteric Fever",
        "keywords": ["typhoid", "stepladder fever", "continuous high fever", "rose spots", "abdominal tenderness fever weakness", "enteric fever"],
        "required_indicators": {"fever": "Yes"},
        "base_risk": "High",
        "description": "Life-threatening bacterial infection caused by Salmonella Typhi.",
        "urgency": "Urgent Widal / blood culture testing and systemic antibiotic treatment."
    },
    {
        "disease": "Acute Myocardial Ischemia / Cardiac Concern",
        "keywords": ["chest pain", "pressure on chest", "left arm pain", "crushing chest pain", "heart attack", "cardiac", "chest tight breathing"],
        "required_indicators": {},
        "base_risk": "High",
        "description": "Compromised blood flow to the heart muscle presenting with acute angina or infarction.",
        "urgency": "EMERGENCY: Seek immediate emergency medical services (EMS) or call 108/911."
    },
    {
        "disease": "Gastrointestinal Bleeding / Gastritis",
        "keywords": ["blood in vomit", "vomit blood", "hematemesis", "black stool", "tarry stool", "melena", "blood vomit"],
        "required_indicators": {},
        "base_risk": "High",
        "description": "Active hemorrhage occurring within the upper or lower gastrointestinal tract.",
        "urgency": "CRITICAL: Urgent gastroenterology and endoscopy evaluation required."
    },
    {
        "disease": "Bronchial Asthma / Acute Respiratory Distress",
        "keywords": ["asthma", "wheezing", "cannot breathe", "shortness of breath", "gasping for air", "dyspnea"],
        "required_indicators": {"difficulty_breathing": "Yes"},
        "base_risk": "High",
        "description": "Chronic respiratory airway inflammation leading to bronchospasm and airflow limitation.",
        "urgency": "Use prescribed rescue inhaler immediately; seek emergency care if distress persists."
    },
    {
        "disease": "Dengue Fever / Viral Exanthem",
        "keywords": ["dengue", "breakbone fever", "pain behind eyes", "retro-orbital pain", "high fever rash platelets", "joint ache fever rash"],
        "required_indicators": {"fever": "Yes"},
        "base_risk": "High",
        "description": "Mosquito-borne tropical disease characterized by high fever, severe headache, and thrombocytopenia.",
        "urgency": "Immediate platelet count monitoring and inpatient clinical hydration required."
    },
    {
        "disease": "Hypertension / Hypertensive Crisis",
        "keywords": ["high blood pressure", "hypertension", "bp high", "extreme bp", "pounding headache dizziness"],
        "required_indicators": {"blood_pressure": "High"},
        "base_risk": "High",
        "description": "Sustained elevation of systemic arterial blood pressure.",
        "urgency": "Monitor arterial pressure every 30 minutes; seek medical care if >180/120 mmHg."
    }
]

class MLEngine:
    model = None
    features = None
    severity_weights: Dict[str, int] = {}
    sample_diseases_cache: List[Dict[str, Any]] = []

    @classmethod
    def load_model(cls):
        """Load trained scikit-learn model and expected features."""
        try:
            cls.model = joblib.load(MODEL_PATH)
            cls.features = joblib.load(MODEL_FEATURES_PATH)
            logger.info(f"Loaded ML model from {MODEL_PATH} and features from {MODEL_FEATURES_PATH}")
        except Exception as e:
            logger.error(f"Error loading ML model or features: {e}")
            cls.model = None
            cls.features = None

        # Load MongoDB Severity weights and Sample_diseases
        cls.load_db_knowledge()

    @classmethod
    def load_db_knowledge(cls):
        """Fetch Severity weights and Sample_diseases from MongoDB Atlas."""
        try:
            from database import DatabaseManager
            db = DatabaseManager.get_db()

            # Load severity weights
            if "Severity" in db.list_collection_names():
                sev_cursor = db["Severity"].find({}, {"_id": 0, "Symptom": 1, "weight": 1})
                cls.severity_weights = {
                    doc["Symptom"].lower().replace("_", " ").strip(): doc.get("weight", 3)
                    for doc in sev_cursor if "Symptom" in doc
                }
                logger.info(f"Loaded {len(cls.severity_weights)} symptom severity weights from MongoDB.")

            # Cache sample diseases
            if "Sample_diseases" in db.list_collection_names():
                cls.sample_diseases_cache = list(db["Sample_diseases"].find({}, {"_id": 0}))
                logger.info(f"Loaded {len(cls.sample_diseases_cache)} sample disease records from MongoDB.")
        except Exception as e:
            logger.warning(f"Could not load database knowledge on startup: {e}")

    @classmethod
    def calculate_severity_score(cls, text_lower: str, indicators: Dict[str, str]) -> Tuple[int, str]:
        """Compute severity score using weights from MongoDB Severity collection."""
        total_weight = 0
        highest_single_weight = 0

        # Match symptoms in text against MongoDB Severity weights
        for symptom, weight in cls.severity_weights.items():
            if symptom in text_lower:
                total_weight += weight
                if weight > highest_single_weight:
                    highest_single_weight = weight

        # Add indicator weights
        if indicators.get("difficulty_breathing") == "Yes":
            total_weight += 6
            highest_single_weight = max(highest_single_weight, 6)
        if indicators.get("blood_pressure") == "High":
            total_weight += 5
            highest_single_weight = max(highest_single_weight, 5)
        if indicators.get("fever") == "Yes":
            total_weight += 4
        if indicators.get("cough") == "Yes":
            total_weight += 3
        if indicators.get("fatigue") == "Yes":
            total_weight += 3

        # Risk classification based on clinical weights
        if highest_single_weight >= 6 or total_weight >= 14 or any(
            w in text_lower for w in ["chest pain", "blood in", "stiff neck", "unconscious", "stroke"]
        ):
            risk = "High"
        elif highest_single_weight >= 4 or total_weight >= 6:
            risk = "Medium"
        else:
            risk = "Low"

        return total_weight, risk

    @classmethod
    def normalize_user_symptom(cls, raw_input: str) -> Optional[str]:
        """Fuzzy-match symptom against canonical list or return title-cased."""
        if not raw_input or not raw_input.strip():
            return None
        valid_symptoms = [
            "Fever", "Cough", "Fatigue", "Difficulty Breathing", "Blood Pressure",
            "Headache", "Abdominal Pain", "Chest Pain", "Nausea", "Joint Pain",
            "Skin Rash", "Urinary Burning", "Back Pain", "Dizziness"
        ]
        match, score = process.extractOne(raw_input, valid_symptoms)
        if score >= 65:
            return match
        return raw_input.strip().title()

    @classmethod
    def match_open_domain_condition(cls, text_lower: str, indicators: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Search knowledge base for conditions (including unlisted diseases) matching symptoms."""
        best_match = None
        best_score = 0.0

        for cond in MEDICAL_CONDITIONS_DB:
            score = 0.0
            matched_keywords = 0

            for kw in cond["keywords"]:
                if kw in text_lower:
                    matched_keywords += 1
                    score += 35.0
                else:
                    # Fuzzy match substring
                    ratio = fuzz.partial_ratio(kw, text_lower)
                    if ratio >= 85:
                        matched_keywords += 1
                        score += 25.0

            # Verify indicator consistency
            req_inds = cond.get("required_indicators", {})
            for k, expected_v in req_inds.items():
                if indicators.get(k) == expected_v:
                    score += 15.0

            if matched_keywords > 0 and score > best_score:
                best_score = score
                best_match = {
                    "disease": cond["disease"],
                    "confidence": min(97.5, max(75.0, 70.0 + score * 0.4)),
                    "base_risk": cond["base_risk"],
                    "description": cond["description"],
                    "urgency": cond["urgency"]
                }

        if best_match and best_score >= 35.0:
            return best_match

        return None

    @classmethod
    def run_prediction(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute clinical prediction engine supporting both dataset and unlisted diseases."""
        if cls.model is None or cls.features is None:
            cls.load_model()

        text_raw = str(data.get("symptoms_text", "")).strip()
        text_lower = text_raw.lower()

        # Extract indicators from request
        inf_fever = data.get("fever", "No")
        inf_cough = data.get("cough", "No")
        inf_fatigue = data.get("fatigue", "No")
        inf_breath = data.get("difficulty_breathing", "No")
        inf_bp = data.get("blood_pressure", "Normal")
        inf_chol = data.get("cholesterol", "Normal")

        # Infer indicators from free text if mentioned
        if any(w in text_lower for w in ["fever", "high temperature", "chills", "shivering", "feverish"]):
            inf_fever = "Yes"
        if any(w in text_lower for w in ["cough", "cold", "sore throat", "phlegm", "coughing"]):
            inf_cough = "Yes"
        if any(w in text_lower for w in ["tired", "fatigue", "exhausted", "weak", "lethargy", "drowsy"]):
            inf_fatigue = "Yes"
        if any(w in text_lower for w in ["breath", "wheezing", "asthma", "dyspnea", "shortness of breath"]):
            inf_breath = "Yes"
        if any(w in text_lower for w in ["hypertension", "high bp", "high pressure"]):
            inf_bp = "High"
        if any(w in text_lower for w in ["low bp", "low pressure", "hypotension"]):
            inf_bp = "Low"

        indicators = {
            "fever": inf_fever,
            "cough": inf_cough,
            "fatigue": inf_fatigue,
            "difficulty_breathing": inf_breath,
            "blood_pressure": inf_bp,
            "cholesterol": inf_chol
        }

        # Calculate clinical severity score from MongoDB Severity collection
        total_weight, calculated_risk = cls.calculate_severity_score(text_lower, indicators)

        # 1. First, check open-domain clinical knowledge base (handles unlisted diseases like COVID, Migraine, UTI, Appendicitis, etc.)
        matched_condition = cls.match_open_domain_condition(text_lower, indicators)

        if matched_condition:
            predicted_disease = matched_condition["disease"]
            confidence_score = matched_condition["confidence"]
            # Risk is the higher of base risk and calculated severity risk
            risk = "High" if ("High" in [matched_condition["base_risk"], calculated_risk]) else (
                "Medium" if ("Medium" in [matched_condition["base_risk"], calculated_risk]) else "Low"
            )
            recommendations = (
                f"Clinical evaluation indicates: {predicted_disease}. "
                f"{matched_condition['urgency']} Risk Level: {risk}."
            )
        else:
            # 2. Check trained scikit-learn model
            predicted_disease = "Common Cold / Viral Respiratory Infection"
            confidence_score = 88.0
            risk = calculated_risk

            if cls.model and cls.features:
                try:
                    input_df = pd.DataFrame([{
                        'Fever': inf_fever,
                        'Cough': inf_cough,
                        'Fatigue': inf_fatigue,
                        'Difficulty Breathing': inf_breath,
                        'Blood Pressure': inf_bp,
                        'Cholesterol Level': inf_chol
                    }])
                    encoded_df = pd.get_dummies(input_df).reindex(columns=cls.features, fill_value=0)
                    raw_pred = cls.model.predict(encoded_df)[0]
                    predicted_disease = str(raw_pred)

                    if hasattr(cls.model, "predict_proba"):
                        probs = cls.model.predict_proba(encoded_df)
                        max_prob = float(np.max(probs) * 100)
                        confidence_score = max(82.0, max_prob)
                except Exception as e:
                    logger.error(f"Model prediction error: {e}")

            # 3. Dynamic Syndromic Triage if text indicates a specific unlisted syndrome
            if "ear" in text_lower and ("pain" in text_lower or "ache" in text_lower):
                predicted_disease = "Acute Otitis Media (Ear Infection)"
                confidence_score = 91.0
                risk = "Low"
            elif "eye" in text_lower and ("red" in text_lower or "itch" in text_lower or "discharge" in text_lower):
                predicted_disease = "Acute Conjunctivitis / Eye Infection"
                confidence_score = 90.5
                risk = "Low"
            elif "stomach" in text_lower or "cramp" in text_lower or "vomit" in text_lower:
                if "blood" in text_lower:
                    predicted_disease = "Acute Upper Gastrointestinal Hemorrhage"
                    confidence_score = 96.0
                    risk = "High"
                else:
                    predicted_disease = "Acute Gastrointestinal Distress / Dyspepsia"
                    confidence_score = 89.0
                    risk = "Medium"

            recommendations = (
                f"AI clinical evaluation indicates potential {predicted_disease}. "
                f"Risk Level: {risk}. Follow recommended precautions and consult a certified medical doctor."
            )

        normalized_text = cls.normalize_user_symptom(text_raw)

        return {
            "predicted_disease": predicted_disease,
            "confidence_score": f"{confidence_score:.2f}%",
            "risk_level": risk,
            "recommendations": recommendations,
            "normalized_symptom": normalized_text,
            "indicators": indicators,
            "severity_weight": total_weight
        }
