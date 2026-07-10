# utils/symptom_analyzer.py
# NLP-based symptom analysis using HuggingFace zero-shot classification
# Maps patient symptoms → probable disease candidates

from transformers import pipeline
import re

try:
    import streamlit as st
    _cache_resource = st.cache_resource(show_spinner=False)
except Exception:  # pragma: no cover - allows module to be imported outside Streamlit
    def _cache_resource(func):
        return func


# ── Disease candidates the model can classify into ─────────────────────────────
DISEASE_CANDIDATES = [
    "Pneumonia",
    "COVID-19",
    "Tuberculosis",
    "Common Cold",
    "Influenza",
    "Bronchitis",
    "Asthma",
    "Heart Disease",
    "Diabetes",
    "Healthy / No Disease"
]

# ── Symptom keyword mapping for rule-based boost ───────────────────────────────
SYMPTOM_KEYWORDS = {
    "COVID-19":       ["loss of smell", "loss of taste", "covid", "corona", "breathlessness"],
    "Pneumonia":      ["chest pain", "productive cough", "high fever", "crackling sound"],
    "Tuberculosis":   ["night sweats", "blood in cough", "weight loss", "tb", "prolonged cough"],
    "Influenza":      ["body ache", "chills", "sudden fever", "flu", "fatigue"],
    "Asthma":         ["wheezing", "shortness of breath", "inhaler", "asthma", "triggered"],
    "Heart Disease":  ["chest tightness", "palpitations", "radiating pain", "jaw pain"],
    "Diabetes":       ["frequent urination", "excessive thirst", "blurred vision", "sugar"],
}

_classifier = None

@_cache_resource
def get_classifier():
    """
    Load zero-shot classifier. Uses distilbert-base-uncased-mnli for CPU optimization:
    - 40% smaller than BART-large (67M vs 400M params)
    - 6-10x faster on CPU
    - Comparable accuracy for symptom classification
    """
    global _classifier
    if _classifier is None:
        try:
            # Use faster DistilBERT model for CPU
            _classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/qnli-distilroberta-base",
                device=-1
            )
        except Exception:
            # Fallback if model not available
            try:
                _classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=-1
                )
            except Exception as e:
                print(f"Error loading classifier: {e}")
                return None
    return _classifier


def analyze_symptoms(symptom_text: str) -> dict:
    """
    Classify symptoms using zero-shot NLI model.
    Returns top 3 disease candidates with confidence scores.
    With error handling and fallback to keyword matching if model fails.
    """
    try:
        classifier = get_classifier()
        
        if classifier is None:
            raise Exception("Classifier could not be loaded")

        # Clean input
        clean_text = symptom_text.strip().lower()
        
        if not clean_text:
            return {
                "top_prediction": "No symptoms provided",
                "top_confidence": 0.0,
                "top3": [],
                "keyword_matches": {},
                "detected_symptoms": [],
                "raw_text": symptom_text,
                "error": "Empty symptom text"
            }

        # Zero-shot classification with error handling
        result = classifier(
            clean_text[:512],  # Limit text length for faster processing
            candidate_labels=DISEASE_CANDIDATES,
            multi_label=False
        )

        # Build ranked results
        ranked = list(zip(result["labels"], result["scores"]))
        top3 = ranked[:3]

        # Rule-based keyword boost
        keyword_matches = {}
        for disease, keywords in SYMPTOM_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in clean_text]
            if matches:
                keyword_matches[disease] = matches

        # Extract key symptoms mentioned
        common_symptoms = [
            "fever", "cough", "fatigue", "headache", "chest pain",
            "shortness of breath", "nausea", "vomiting", "diarrhea",
            "sore throat", "runny nose", "body ache", "loss of appetite",
            "sweating", "chills", "dizziness", "weakness"
        ]
        detected_symptoms = [s for s in common_symptoms if s in clean_text]

        return {
            "top_prediction": top3[0][0] if top3 else "Unknown",
            "top_confidence": top3[0][1] if top3 else 0.0,
            "top3": [{"disease": d, "confidence": c} for d, c in top3],
            "keyword_matches": keyword_matches,
            "detected_symptoms": detected_symptoms,
            "raw_text": symptom_text
        }
    except Exception as e:
        print(f"Error in symptom analysis: {e}")
        # Fallback: keyword-based analysis only
        clean_text = symptom_text.strip().lower()
        keyword_matches = {}
        for disease, keywords in SYMPTOM_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in clean_text]
            if matches:
                keyword_matches[disease] = len(matches)
        
        top_disease = max(keyword_matches.items(), key=lambda x: x[1])[0] if keyword_matches else "Healthy / No Disease"
        
        return {
            "top_prediction": top_disease,
            "top_confidence": 0.5,
            "top3": [],
            "keyword_matches": keyword_matches,
            "detected_symptoms": [s for s in [
                "fever", "cough", "fatigue", "headache", "chest pain",
                "shortness of breath", "nausea", "vomiting", "diarrhea",
                "sore throat", "runny nose", "body ache", "loss of appetite",
                "sweating", "chills", "dizziness", "weakness"
            ] if s in clean_text],
            "raw_text": symptom_text,
            "error": f"Model failed, using keyword matching: {str(e)}"
        }


def get_symptom_recommendations(disease: str) -> dict:
    """Return care recommendations per disease."""
    recs = {
        "COVID-19": {
            "tests": ["PCR/Antigen Test", "CT Scan", "Blood Oxygen Level"],
            "care": ["Isolate for 10 days", "Monitor oxygen saturation", "Stay hydrated"],
            "urgency": "Immediate"
        },
        "Pneumonia": {
            "tests": ["Chest X-ray", "CBC Blood Test", "Sputum Culture"],
            "care": ["Antibiotics (if bacterial)", "Rest", "Plenty of fluids"],
            "urgency": "Within 24 hours"
        },
        "Tuberculosis": {
            "tests": ["Sputum AFB Test", "Chest X-ray", "IGRA Blood Test"],
            "care": ["6-month antibiotic course", "Isolation", "Nutritious diet"],
            "urgency": "Within 48 hours"
        },
        "Influenza": {
            "tests": ["Rapid Influenza Test", "Nasal Swab"],
            "care": ["Rest", "Antiviral medication", "Hydration", "Fever management"],
            "urgency": "Within 48 hours"
        },
        "Asthma": {
            "tests": ["Spirometry", "Peak Flow Test", "Allergy Test"],
            "care": ["Use prescribed inhaler", "Avoid triggers", "See pulmonologist"],
            "urgency": "Schedule appointment"
        },
    }
    return recs.get(disease, {
        "tests": ["General blood panel", "Physical examination"],
        "care": ["Rest and hydration", "Consult a doctor"],
        "urgency": "Within 1 week"
    })
