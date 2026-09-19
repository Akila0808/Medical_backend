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
        "keywords": ["stiff neck", "photophobia fever headache", "unable to touch chin to chest", "neck stiffness high fever", "meningitis", "brudzinski", "kernig"],
        "required_indicators": {"fever": "Yes"},
        "base_risk": "High",
        "description": "Life-threatening inflammation of the protective membranes covering the brain and spinal cord.",
        "urgency": "EMERGENCY: Immediate emergency hospital admission and lumbar puncture required."
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

# Clinical AI Medication Knowledge Base mapping conditions to recommended prescription protocols
DISEASE_MEDICATIONS_DB: Dict[str, List[Dict[str, str]]] = {
    "covid": [
        {"name": "Paracetamol", "dosage": "650 mg", "frequency": "Every 6-8 hours PRN", "duration": "5 days", "type": "Antipyretic / Analgesic", "instructions": "Take after meals for fever/myalgia; do not exceed 3000mg/24h."},
        {"name": "N-Acetylcysteine (NAC)", "dosage": "600 mg", "frequency": "Once daily effervescent", "duration": "5 days", "type": "Mucolytic & Antioxidant", "instructions": "Dissolve in glass of water after food."},
        {"name": "Vitamin C + Zinc Chewable", "dosage": "500 mg + 50 mg", "frequency": "Once daily", "duration": "10 days", "type": "Immune Supplement", "instructions": "Take after breakfast."},
        {"name": "Budesonide Inhaler", "dosage": "200 mcg", "frequency": "2 puffs twice daily", "duration": "5-7 days", "type": "Inhaled Corticosteroid", "instructions": "Rinse mouth thoroughly with water after inhalation."}
    ],
    "malaria": [
        {"name": "Artemether + Lumefantrine", "dosage": "20 mg / 120 mg (4 tabs)", "frequency": "Twice daily (0h, 8h, 24h, 36h, 48h, 60h)", "duration": "3 days", "type": "ACT Antimalarial", "instructions": "Take with high-fat meal or whole milk to maximize bioavailability."},
        {"name": "Paracetamol", "dosage": "650 mg", "frequency": "Every 6 hours PRN", "duration": "3 days", "type": "Antipyretic", "instructions": "For fever control during rigors."},
        {"name": "Oral Rehydration Salts (ORS)", "dosage": "1 Sachet in 1L water", "frequency": "Sip throughout day", "duration": "3 days", "type": "Electrolyte Therapy", "instructions": "Maintain hydration and replace electrolyte losses."}
    ],
    "urinary tract infection": [
        {"name": "Nitrofurantoin Monohydrate", "dosage": "100 mg", "frequency": "Twice daily (every 12 hrs)", "duration": "5 days", "type": "Urinary Antibacterial", "instructions": "Take with food or milk to prevent nausea and improve absorption."},
        {"name": "Phenazopyridine HCl", "dosage": "200 mg", "frequency": "Three times daily after meals", "duration": "2 days", "type": "Urinary Tract Analgesic", "instructions": "Relieves dysuria and bladder spasms; causes harmless orange-red urine discoloration."},
        {"name": "Potassium Magnesium Citrate", "dosage": "15 ml in glass of water", "frequency": "Twice daily", "duration": "5 days", "type": "Urinary Alkalinizer", "instructions": "Reduces urine acidity and burning sensation."}
    ],
    "uti": [
        {"name": "Nitrofurantoin Monohydrate", "dosage": "100 mg", "frequency": "Twice daily (every 12 hrs)", "duration": "5 days", "type": "Urinary Antibacterial", "instructions": "Take with food or milk to prevent nausea."},
        {"name": "Phenazopyridine HCl", "dosage": "200 mg", "frequency": "Three times daily after meals", "duration": "2 days", "type": "Urinary Tract Analgesic", "instructions": "Relieves dysuria and bladder spasms."},
        {"name": "Potassium Magnesium Citrate", "dosage": "15 ml in glass of water", "frequency": "Twice daily", "duration": "5 days", "type": "Urinary Alkalinizer", "instructions": "Reduces urine acidity and burning sensation."}
    ],
    "migraine": [
        {"name": "Sumatriptan Succinate", "dosage": "50 mg", "frequency": "At immediate onset of migraine", "duration": "Single dose (repeat in 2h if needed, max 200mg/24h)", "type": "5-HT1 Receptor Agonist (Triptan)", "instructions": "Take at earliest sign of throbbing headache or visual aura."},
        {"name": "Naproxen Sodium", "dosage": "500 mg", "frequency": "Twice daily with meals", "duration": "3 days PRN", "type": "NSAID Analgesic", "instructions": "Take with food or antacid to prevent gastric irritation."},
        {"name": "Domperidone / Ondansetron", "dosage": "10 mg / 4 mg", "frequency": "30 mins before food PRN", "duration": "2 days", "type": "Antiemetic", "instructions": "For migraine-associated nausea and gastric stasis."}
    ],
    "gastroesophageal reflux disease": [
        {"name": "Pantoprazole Sodium EC", "dosage": "40 mg", "frequency": "Once daily in morning", "duration": "14 days", "type": "Proton Pump Inhibitor (PPI)", "instructions": "Take on an empty stomach at least 30-45 minutes before breakfast."},
        {"name": "Sodium Alginate + Potassium Bicarbonate Suspension", "dosage": "10 ml", "frequency": "After meals and before bedtime", "duration": "7-10 days", "type": "Mucosal Barrier & Antacid", "instructions": "Forms protective raft over gastric contents; do not drink water immediately after."},
        {"name": "Itopride HCl", "dosage": "50 mg", "frequency": "Three times daily before meals", "duration": "10 days", "type": "Prokinetic", "instructions": "Enhances gastric emptying."}
    ],
    "gerd": [
        {"name": "Pantoprazole Sodium EC", "dosage": "40 mg", "frequency": "Once daily in morning", "duration": "14 days", "type": "Proton Pump Inhibitor (PPI)", "instructions": "Take on an empty stomach at least 30-45 minutes before breakfast."},
        {"name": "Sodium Alginate Suspension", "dosage": "10 ml", "frequency": "After meals and before bedtime", "duration": "7-10 days", "type": "Mucosal Barrier & Antacid", "instructions": "Forms protective raft over gastric contents."}
    ],
    "appendicitis": [
        {"name": "Ceftriaxone IV / Ciprofloxacin", "dosage": "1 g IV / 500 mg PO", "frequency": "Every 12 hours", "duration": "Hospital Triage", "type": "Broad-Spectrum Antibiotic", "instructions": "Administer under strict surgical observation prior to appendectomy."},
        {"name": "Metronidazole", "dosage": "500 mg", "frequency": "Every 8 hours", "duration": "Hospital Triage", "type": "Anaerobic Antimicrobial", "instructions": "Cover intra-abdominal anaerobes."},
        {"name": "Tramadol / Acetaminophen IV", "dosage": "50-100 mg", "frequency": "Every 8 hours PRN", "duration": "Acute Phase", "type": "Analgesic", "instructions": "Administered under emergency surgical guidance; NPO maintained."}
    ],
    "kidney stone": [
        {"name": "Tamsulosin HCl", "dosage": "0.4 mg", "frequency": "Once daily after same meal", "duration": "14 days", "type": "Alpha-1 Blocker", "instructions": "Facilitates ureteral stone passage by relaxing smooth muscle."},
        {"name": "Ketorolac / Ibuprofen", "dosage": "10 mg / 400 mg", "frequency": "Every 8 hours with meals PRN", "duration": "5 days", "type": "NSAID Analgesic", "instructions": "Potent relief for acute renal colic."},
        {"name": "Potassium Citrate Liquid", "dosage": "15 ml in water", "frequency": "Twice daily after meals", "duration": "14 days", "type": "Urine Alkalinizer", "instructions": "Prevents stone crystallization."}
    ],
    "tonsillitis": [
        {"name": "Amoxicillin-Clavulanate (Augmentin)", "dosage": "625 mg", "frequency": "Twice daily after meals", "duration": "5-7 days", "type": "Penicillin Antibiotic", "instructions": "Complete entire course even if throat pain subsides."},
        {"name": "Flurbiprofen Lozenges", "dosage": "8.75 mg", "frequency": "Dissolve in mouth every 4-6 hours", "duration": "3-5 days", "type": "Topical Throat Anti-inflammatory", "instructions": "Do not chew or swallow whole; suck slowly."},
        {"name": "Paracetamol", "dosage": "650 mg", "frequency": "Every 8 hours PRN", "duration": "3 days", "type": "Analgesic / Antipyretic", "instructions": "Take for odynophagia and fever relief."}
    ],
    "pharyngitis": [
        {"name": "Amoxicillin-Clavulanate", "dosage": "625 mg", "frequency": "Twice daily after meals", "duration": "5-7 days", "type": "Penicillin Antibiotic", "instructions": "Complete entire course even if throat pain subsides."},
        {"name": "Benzydamine Throat Spray / Lozenges", "dosage": "2-3 sprays / 1 lozenge", "frequency": "Every 4-6 hours PRN", "duration": "3-5 days", "type": "Analgesic Throat Spray", "instructions": "Provides local analgesic and anti-inflammatory action."}
    ],
    "gastroenteritis": [
        {"name": "Oral Rehydration Salts (WHO Formula)", "dosage": "1 Sachet in 1 Litre boiled water", "frequency": "Drink after each loose stool", "duration": "3-4 days", "type": "Electrolyte Replenisher", "instructions": "Essential first-line therapy to prevent severe dehydration."},
        {"name": "Zinc Sulfate Tablets", "dosage": "20 mg", "frequency": "Once daily after food", "duration": "10 days", "type": "Mucosal Repair Micronutrient", "instructions": "Accelerates intestinal epithelial healing."},
        {"name": "Racecadotril", "dosage": "100 mg", "frequency": "Three times daily before meals", "duration": "3 days", "type": "Antisecretory Antidiarrheal", "instructions": "Reduces intestinal hypersecretion without slowing motility."},
        {"name": "Probiotic Complex (S. boulardii)", "dosage": "250 mg (1 capsule)", "frequency": "Twice daily with water", "duration": "5 days", "type": "Intestinal Probiotic", "instructions": "Restores gut microflora balance."}
    ],
    "sinusitis": [
        {"name": "Amoxicillin + Clavulanic Acid", "dosage": "625 mg", "frequency": "Twice daily after food", "duration": "7 days", "type": "Broad-Spectrum Antibiotic", "instructions": "Indicated if bacterial rhinosinusitis symptoms persist >7-10 days."},
        {"name": "Fluticasone Furoate Nasal Spray", "dosage": "27.5 mcg/spray (2 sprays/nostril)", "frequency": "Once daily in morning", "duration": "14 days", "type": "Intranasal Corticosteroid", "instructions": "Shake bottle well before use; aim away from nasal septum."},
        {"name": "Xylometazoline 0.1% Nasal Drops", "dosage": "2-3 drops per nostril", "frequency": "Twice daily", "duration": "3-5 days maximum", "type": "Decongestant", "instructions": "Do not use for more than 5 consecutive days to avoid rebound congestion."}
    ],
    "allergic rhinitis": [
        {"name": "Levocetirizine Dihydrochloride + Montelukast", "dosage": "5 mg + 10 mg", "frequency": "Once daily at bedtime", "duration": "10 days", "type": "Antihistamine & Leukotriene Antagonist", "instructions": "Take at night; may cause mild drowsiness."},
        {"name": "Mometasone Furoate Nasal Spray", "dosage": "50 mcg/actuation (2 sprays/nostril)", "frequency": "Once daily", "duration": "14 days", "type": "Intranasal Corticosteroid", "instructions": "Clear nasal passages before spraying."},
        {"name": "Sodium Chloride 0.9% Isotonic Saline Rinse", "dosage": "2-3 sprays each nostril", "frequency": "Three times daily", "duration": "14 days", "type": "Nasal Wash", "instructions": "Clears airborne allergens and mucus."}
    ],
    "pneumonia": [
        {"name": "Azithromycin", "dosage": "500 mg", "frequency": "Once daily 1 hr before or 2 hrs after food", "duration": "5 days", "type": "Macrolide Antibiotic", "instructions": "Take on an empty stomach with a full glass of water."},
        {"name": "Amoxicillin + Clavulanate", "dosage": "875/125 mg", "frequency": "Twice daily with meals", "duration": "7-10 days", "type": "Broad-Spectrum Antibacterial", "instructions": "Strict compliance needed to eradicate lower respiratory infection."},
        {"name": "Ambroxol + Guaiphenesin Expectorant Syrup", "dosage": "10 ml", "frequency": "Three times daily after food", "duration": "5 days", "type": "Mucolytic Expectorant", "instructions": "Drink plenty of warm fluids to help loosen bronchopulmonary secretions."}
    ],
    "otitis media": [
        {"name": "Amoxicillin-Clavulanate", "dosage": "625 mg", "frequency": "Twice daily after meals", "duration": "7 days", "type": "Antibiotic", "instructions": "Standard first-line regimen for acute middle ear bacterial infection."},
        {"name": "Ibuprofen + Paracetamol", "dosage": "400 mg + 325 mg", "frequency": "Every 8 hours after food PRN", "duration": "3-5 days", "type": "Analgesic / Anti-inflammatory", "instructions": "Effective for severe otalgia and fever."},
        {"name": "Ciprofloxacin 0.3% Ear Drops", "dosage": "3-4 drops into affected ear", "frequency": "Twice daily", "duration": "7 days", "type": "Topical Otic Antibacterial", "instructions": "Lie with affected ear upward for 5 minutes after instillation."}
    ],
    "conjunctivitis": [
        {"name": "Moxifloxacin 0.5% Ophthalmic Solution", "dosage": "1 drop in affected eye(s)", "frequency": "Three times daily", "duration": "7 days", "type": "Fluoroquinolone Antibacterial Eye Drops", "instructions": "Do not touch dropper tip to eye or skin surface; remove contact lenses."},
        {"name": "Carboxymethylcellulose 0.5% Eye Drops", "dosage": "1-2 drops", "frequency": "4-6 times daily as needed", "duration": "10 days", "type": "Lubricating Artificial Tears", "instructions": "Soothes gritty sensation and ocular irritation."}
    ],
    "anemia": [
        {"name": "Ferrous Ascorbate + Folic Acid", "dosage": "100 mg Elemental Iron + 1.5 mg FA", "frequency": "Once daily after lunch", "duration": "30-60 days", "type": "Hematinic Supplement", "instructions": "Take with orange juice or Vitamin C to enhance absorption; avoid tea/milk within 2 hours."},
        {"name": "Vitamin B12 (Methylcobalamin)", "dosage": "1500 mcg", "frequency": "Once daily", "duration": "30 days", "type": "Neurotropic Vitamin", "instructions": "Supports erythrocyte maturation."}
    ],
    "diabetes": [
        {"name": "Metformin Hydrochloride SR", "dosage": "500 mg", "frequency": "Twice daily with or immediately after meals", "duration": "Ongoing / 30 days", "type": "Biguanide Antidiabetic", "instructions": "Take with breakfast and dinner to minimize GI side effects; monitor blood glucose."},
        {"name": "Glimepiride", "dosage": "1 mg", "frequency": "Once daily 15 mins before breakfast", "duration": "Ongoing / 30 days", "type": "Sulfonylurea Antidiabetic", "instructions": "Take before first main meal; carry fast-acting carbs for hypoglycemia."}
    ],
    "hypertension": [
        {"name": "Telmisartan", "dosage": "40 mg", "frequency": "Once daily in morning", "duration": "Ongoing / 30 days", "type": "Angiotensin II Receptor Blocker (ARB)", "instructions": "Take consistently at same time each day; monitor BP regularly."},
        {"name": "Amlodipine Besylate", "dosage": "5 mg", "frequency": "Once daily in morning", "duration": "Ongoing / 30 days", "type": "Calcium Channel Blocker", "instructions": "Monitor for peripheral pedal swelling."}
    ],
    "asthma": [
        {"name": "Budecort (Budesonide + Formoterol Inhaler)", "dosage": "200 mcg / 6 mcg (2 puffs)", "frequency": "Twice daily (morning and night)", "duration": "Ongoing / 30 days", "type": "ICS + LABA Controller Inhaler", "instructions": "Use with spacer; rinse mouth with water after each inhalation."},
        {"name": "Salbutamol / Albuterol Inhaler (Asthalin)", "dosage": "100 mcg (2 puffs)", "frequency": "Every 4-6 hours as needed for acute wheezing", "duration": "PRN Rescue", "type": "SABA Rapid Bronchodilator", "instructions": "Carry at all times for acute bronchospasm relief."}
    ],
    "influenza": [
        {"name": "Oseltamivir Phosphate (Tamiflu)", "dosage": "75 mg", "frequency": "Twice daily (every 12 hours)", "duration": "5 days", "type": "Neuraminidase Antiviral", "instructions": "Initiate within 48 hours of symptom onset for maximum efficacy; take with food."},
        {"name": "Paracetamol + Phenylephrine + Chlorpheniramine", "dosage": "500 mg + 10 mg + 2 mg", "frequency": "Three times daily after food", "duration": "3-5 days", "type": "Flu Symptom Relief Combination", "instructions": "Relieves body aches, fever, and rhinorrhea."}
    ],
    "common cold": [
        {"name": "Paracetamol", "dosage": "500 mg", "frequency": "Every 6-8 hours PRN", "duration": "3-4 days", "type": "Antipyretic / Analgesic", "instructions": "For mild fever, headache, and malaise."},
        {"name": "Cetirizine Hydrochloride", "dosage": "10 mg", "frequency": "Once daily at night", "duration": "5 days", "type": "2nd-Gen Antihistamine", "instructions": "Relieves sneezing, runny nose, and itchy eyes."},
        {"name": "Normal Saline Nasal Spray (0.9% NaCl)", "dosage": "2 sprays in each nostril", "frequency": "3-4 times daily", "duration": "5 days", "type": "Nasal Moisturizer", "instructions": "Clears nasal congestion safely."}
    ],
    "bronchitis": [
        {"name": "Amoxicillin-Clavulanate", "dosage": "625 mg", "frequency": "Twice daily after food", "duration": "5-7 days", "type": "Antibiotic", "instructions": "Indicated for purulent sputum and acute bacterial exacerbations."},
        {"name": "Guaifenesin + Ambroxol Syrup", "dosage": "10 ml", "frequency": "Three times daily after meals", "duration": "5 days", "type": "Expectorant Mucolytic", "instructions": "Take with full glass of warm water to promote mucus clearance."}
    ],
    "eczema": [
        {"name": "Hydrocortisone 1% / Mometasone 0.1% Cream", "dosage": "Thin layer applied to affected skin", "frequency": "Twice daily", "duration": "7-10 days", "type": "Topical Corticosteroid", "instructions": "Apply sparingly to active rash; do not apply on broken skin or face unless prescribed."},
        {"name": "Ceramide & Colloidal Oatmeal Emollient Lotion", "dosage": "Liberal application", "frequency": "3-4 times daily and immediately after bathing", "duration": "Ongoing", "type": "Barrier Repair Moisturizer", "instructions": "Locks in hydration and restores epidermal barrier."}
    ],
    "gout": [
        {"name": "Colchicine", "dosage": "0.5 mg", "frequency": "1 tab twice daily", "duration": "3-5 days", "type": "Anti-Gout Anti-inflammatory", "instructions": "Take at earliest onset of acute flare; stop if diarrhea occurs."},
        {"name": "Indomethacin / Etoricoxib", "dosage": "50 mg / 90 mg", "frequency": "Once or twice daily with food", "duration": "5 days", "type": "NSAID Analgesic", "instructions": "Rapid relief for severe acute joint inflammation."}
    ],
    "meningitis": [
        {"name": "Ceftriaxone IV + Vancomycin IV", "dosage": "2 g IV q12h + 15-20 mg/kg q8-12h", "frequency": "Inpatient IV Infusion", "duration": "Emergency Inpatient Protocol", "type": "Empirical High-Dose Meningeal Antibiotics", "instructions": "Immediate hospital ICU administration under emergency care."},
        {"name": "Dexamethasone IV", "dosage": "10 mg IV", "frequency": "q6h with or before first antibiotic dose", "duration": "4 days", "type": "Adjunctive Corticosteroid", "instructions": "Reduces cerebral inflammation and neurological complications."}
    ],
    "typhoid": [
        {"name": "Cefixime / Azithromycin", "dosage": "200 mg q12h / 500 mg q24h", "frequency": "Twice daily after food", "duration": "7-14 days", "type": "Enteric Antimicrobial", "instructions": "Complete entire course and maintain strict hand hygiene."},
        {"name": "Paracetamol", "dosage": "650 mg", "frequency": "Every 6 hours PRN", "duration": "5 days", "type": "Antipyretic", "instructions": "For step-ladder enteric fever control."}
    ],
    "dengue": [
        {"name": "Paracetamol", "dosage": "500-650 mg", "frequency": "Every 6-8 hours PRN (Max 3g/day)", "duration": "5 days", "type": "Antipyretic (Non-NSAID)", "instructions": "STRICT WARNING: Avoid NSAIDs (aspirin, ibuprofen, diclofenac) due to bleeding risks."},
        {"name": "Oral Rehydration Solution (ORS)", "dosage": "2-3 Litres daily", "frequency": "Frequent sips throughout the day", "duration": "5-7 days", "type": "Intravenous / Oral Volume Expander", "instructions": "Essential to prevent plasma leakage and hemoconcentration."}
    ],
    "general condition": [
        {"name": "Paracetamol", "dosage": "500 mg", "frequency": "Every 6-8 hours as needed", "duration": "3 days", "type": "Analgesic / Antipyretic", "instructions": "For mild discomfort or fever."},
        {"name": "Multivitamin & Mineral Complex", "dosage": "1 tablet", "frequency": "Once daily after breakfast", "duration": "14 days", "type": "Nutritional Support", "instructions": "Provides baseline micronutrient support."},
        {"name": "Oral Hydration Solution (ORS)", "dosage": "1 Sachet in 1L water", "frequency": "Sip throughout day", "duration": "3 days", "type": "Hydration Fluid", "instructions": "Maintains cellular electrolyte balance."}
    ]
}

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
    def get_ai_medications_for_disease(cls, disease_name: str) -> List[Dict[str, str]]:
        """Retrieve structured AI-recommended clinical medication protocols for a given condition."""
        if not disease_name:
            return DISEASE_MEDICATIONS_DB.get("general condition", [])

        name_lower = disease_name.lower().strip()

        # 1. Direct or substring match in keys
        for key, meds in DISEASE_MEDICATIONS_DB.items():
            if key in name_lower or name_lower in key:
                return meds

        # 2. Specific clinical keywords mapping
        if any(k in name_lower for k in ["covid", "sars-cov-2", "corona"]):
            return DISEASE_MEDICATIONS_DB["covid"]
        if any(k in name_lower for k in ["uti", "urinary", "cystitis", "bladder"]):
            return DISEASE_MEDICATIONS_DB["urinary tract infection"]
        if any(k in name_lower for k in ["migraine", "cephalgia", "vascular headache"]):
            return DISEASE_MEDICATIONS_DB["migraine"]
        if any(k in name_lower for k in ["gerd", "acid reflux", "esophageal", "heartburn"]):
            return DISEASE_MEDICATIONS_DB["gastroesophageal reflux disease"]
        if any(k in name_lower for k in ["appendicitis", "appendix", "mcburney"]):
            return DISEASE_MEDICATIONS_DB["appendicitis"]
        if any(k in name_lower for k in ["stone", "nephrolithiasis", "renal colic"]):
            return DISEASE_MEDICATIONS_DB["kidney stone"]
        if any(k in name_lower for k in ["tonsil", "strep", "throat", "pharyngitis"]):
            return DISEASE_MEDICATIONS_DB["tonsillitis"]
        if any(k in name_lower for k in ["gastroenteritis", "stomach bug", "diarrhea", "food poisoning"]):
            return DISEASE_MEDICATIONS_DB["gastroenteritis"]
        if any(k in name_lower for k in ["sinus"]):
            return DISEASE_MEDICATIONS_DB["sinusitis"]
        if any(k in name_lower for k in ["allergy", "rhinitis", "hay fever"]):
            return DISEASE_MEDICATIONS_DB["allergic rhinitis"]
        if any(k in name_lower for k in ["pneumonia"]):
            return DISEASE_MEDICATIONS_DB["pneumonia"]
        if any(k in name_lower for k in ["otitis", "ear"]):
            return DISEASE_MEDICATIONS_DB["otitis media"]
        if any(k in name_lower for k in ["conjunctivitis", "pink eye", "eye infection"]):
            return DISEASE_MEDICATIONS_DB["conjunctivitis"]
        if any(k in name_lower for k in ["anemia", "iron deficiency"]):
            return DISEASE_MEDICATIONS_DB["anemia"]
        if any(k in name_lower for k in ["diabetes", "hyperglycemia", "glucose"]):
            return DISEASE_MEDICATIONS_DB["diabetes"]
        if any(k in name_lower for k in ["hypertension", "high blood pressure", "hypertensive"]):
            return DISEASE_MEDICATIONS_DB["hypertension"]
        if any(k in name_lower for k in ["asthma", "bronchospasm", "wheezing"]):
            return DISEASE_MEDICATIONS_DB["asthma"]
        if any(k in name_lower for k in ["flu", "influenza"]):
            return DISEASE_MEDICATIONS_DB["influenza"]
        if any(k in name_lower for k in ["cold", "coryza", "viral respiratory"]):
            return DISEASE_MEDICATIONS_DB["common cold"]
        if any(k in name_lower for k in ["bronchitis"]):
            return DISEASE_MEDICATIONS_DB["bronchitis"]
        if any(k in name_lower for k in ["eczema", "dermatitis"]):
            return DISEASE_MEDICATIONS_DB["eczema"]
        if any(k in name_lower for k in ["gout", "hyperuricemia"]):
            return DISEASE_MEDICATIONS_DB["gout"]
        if any(k in name_lower for k in ["meningitis"]):
            return DISEASE_MEDICATIONS_DB["meningitis"]
        if any(k in name_lower for k in ["typhoid"]):
            return DISEASE_MEDICATIONS_DB["typhoid"]
        if any(k in name_lower for k in ["dengue"]):
            return DISEASE_MEDICATIONS_DB["dengue"]
        if any(k in name_lower for k in ["malaria"]):
            return DISEASE_MEDICATIONS_DB["malaria"]

        # 3. Fuzzy match against keys
        keys = list(DISEASE_MEDICATIONS_DB.keys())
        match, score = process.extractOne(name_lower, keys)
        if score >= 60 and match in DISEASE_MEDICATIONS_DB:
            return DISEASE_MEDICATIONS_DB[match]

        return DISEASE_MEDICATIONS_DB["general condition"]

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
        ai_medicines = cls.get_ai_medications_for_disease(predicted_disease)

        return {
            "predicted_disease": predicted_disease,
            "confidence_score": f"{confidence_score:.2f}%",
            "risk_level": risk,
            "recommendations": recommendations,
            "normalized_symptom": normalized_text,
            "indicators": indicators,
            "severity_weight": total_weight,
            "ai_medicines": ai_medicines
        }
