from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
import joblib
import traceback
import numpy as np
import pandas as pd

from nlp.preprocess import normalize_text
from nlp.extractor import (
    extract_symptoms,
    extract_duration,
    extract_severity_words
)
from nlp.rules import (
    detect_negations,
    detect_danger_terms,
    adjust_severity_with_rules,
    generate_recommendation
)


app = FastAPI(title="SACA Medical Triage API")

# Allow requests from your frontend during demo hosting.
# For a production system, replace ["*"] with your exact frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request / Response Models
# ============================================================

class PredictRequest(BaseModel):
    text: str
    age: Optional[int] = None
    gender: Optional[str] = None
    pain_score: Optional[int] = Field(default=None, ge=0, le=10)
    body_part: Optional[str] = None


class DiseasePrediction(BaseModel):
    name: str
    probability: float


class PredictResponse(BaseModel):
    cleaned_text: str
    severity: str
    confidence: Optional[float]
    possible_diseases: List[DiseasePrediction]
    symptoms: List[str]
    duration: Optional[str]
    severity_words: List[str]
    danger_terms: List[str]
    negations: List[str]
    recommendation: str
    pain_score: Optional[int] = None
    body_part: Optional[str] = None
    disclaimer: str


# ============================================================
# File Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Your current severity model path
SEVERITY_MODEL_PATH = (
    BASE_DIR / "ml" / "saved" / "linear_svc.joblib"
)

# Your disease model path
DISEASE_MODEL_PATH = (
    BASE_DIR / "ml" / "saved" / "disease" / "disease_model.joblib"
)


# ============================================================
# Global Models
# ============================================================

severity_model = None
severity_vectorizer = None

disease_model = None
disease_vectorizer = None


# ============================================================
# Model Loading Helpers
# ============================================================

def unwrap_model_and_vectorizer(loaded_object, model_name: str):
    """
    Handles two save formats.

    Format 1:
        joblib.dump(pipeline, path)

    Format 2:
        joblib.dump({
            "model": model,
            "vectorizer": vectorizer
        }, path)

    Returns:
        model, vectorizer

    If the object is already a sklearn Pipeline, vectorizer is None.
    """

    if not isinstance(loaded_object, dict):
        return loaded_object, None

    print(f"\n{model_name} file is a dictionary.")
    print(f"{model_name} keys:", list(loaded_object.keys()))

    model_keys = [
        "model",
        "pipeline",
        "linear_svc",
        "severity_model",
        "disease_model",
        "best_model",
        "classifier",
        "clf"
    ]

    vectorizer_keys = [
        "vectorizer",
        "tfidf",
        "tfidf_vectorizer",
        "severity_vectorizer",
        "disease_vectorizer"
    ]

    found_model = None
    found_vectorizer = None

    for key in model_keys:
        if key in loaded_object:
            found_model = loaded_object[key]
            break

    for key in vectorizer_keys:
        if key in loaded_object:
            found_vectorizer = loaded_object[key]
            break

    if found_model is None:
        for value in loaded_object.values():
            if hasattr(value, "predict"):
                found_model = value
                break

    if found_vectorizer is None:
        for value in loaded_object.values():
            if hasattr(value, "transform") and not hasattr(value, "predict"):
                found_vectorizer = value
                break

    if found_model is None:
        raise RuntimeError(
            f"No valid model found inside {model_name} dictionary. "
            f"Available keys: {list(loaded_object.keys())}"
        )

    return found_model, found_vectorizer


def load_models():
    global severity_model, severity_vectorizer
    global disease_model, disease_vectorizer

    if not SEVERITY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Severity model not found: {SEVERITY_MODEL_PATH}\n"
            "Check your saved model path or run your severity training script."
        )

    if not DISEASE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Disease model not found: {DISEASE_MODEL_PATH}\n"
            "Run: python ml/models/disease/compare_disease_models.py"
        )

    loaded_severity = joblib.load(SEVERITY_MODEL_PATH)
    loaded_disease = joblib.load(DISEASE_MODEL_PATH)

    severity_model, severity_vectorizer = unwrap_model_and_vectorizer(
        loaded_severity,
        "Severity model"
    )

    disease_model, disease_vectorizer = unwrap_model_and_vectorizer(
        loaded_disease,
        "Disease model"
    )

    print("\n✅ Models loaded successfully")
    print("Severity model path:", SEVERITY_MODEL_PATH)
    print("Disease model path:", DISEASE_MODEL_PATH)

    print("\nModel types:")
    print("Severity model:", type(severity_model))
    print("Severity vectorizer:", type(severity_vectorizer))
    print("Disease model:", type(disease_model))
    print("Disease vectorizer:", type(disease_vectorizer))

    print("\nCapability check:")
    print("Severity has predict:", hasattr(severity_model, "predict"))
    print("Severity has predict_proba:", hasattr(severity_model, "predict_proba"))
    print("Disease has predict:", hasattr(disease_model, "predict"))
    print("Disease has predict_proba:", hasattr(disease_model, "predict_proba"))


try:
    load_models()
except Exception:
    print("\n❌ Error loading models:")
    print(traceback.format_exc())


def get_severity_model():
    if severity_model is None:
        raise RuntimeError(
            "Severity model is not loaded. "
            "Check /health and confirm the severity model path."
        )

    return severity_model


def get_disease_model():
    if disease_model is None:
        raise RuntimeError(
            "Disease model is not loaded. "
            "Check /health and confirm the disease model path."
        )

    return disease_model


# ============================================================
# Severity Prediction
# ============================================================

def map_severity_label(prediction):
    label_map = {
        0: "mild",
        1: "moderate",
        2: "severe",
        "0": "mild",
        "1": "moderate",
        "2": "severe",
        "mild": "mild",
        "moderate": "moderate",
        "severe": "severe",
    }

    prediction = str(prediction).lower().strip()

    return label_map.get(prediction, prediction)


def predict_severity(cleaned_text: str):
    """
    Supports:
    1. Full sklearn Pipeline:
       pipeline.predict([cleaned_text])

    2. Separate vectorizer + model:
       vectorizer.transform([cleaned_text])
       model.predict(X)
    """

    model = get_severity_model()

    if severity_vectorizer is not None:
        X = severity_vectorizer.transform([cleaned_text])
        raw_prediction = model.predict(X)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            confidence = float(max(probabilities))

    else:
        raw_prediction = model.predict([cleaned_text])[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([cleaned_text])[0]
            confidence = float(max(probabilities))

    severity = map_severity_label(raw_prediction)

    return severity, confidence


# ============================================================
# Disease Prediction Helpers
# ============================================================

def get_age_group(age: Optional[int]) -> str:
    if age is None:
        return "adult"

    if age < 16:
        return "child"

    if age >= 65:
        return "older_adult"

    return "adult"


def estimate_duration_days(duration_text: Optional[str]) -> int:
    if not duration_text:
        return 1

    duration_text = str(duration_text).lower()

    if (
        "today" in duration_text
        or "this morning" in duration_text
        or "right now" in duration_text
        or "suddenly" in duration_text
    ):
        return 0

    if "yesterday" in duration_text:
        return 1

    if "few days" in duration_text:
        return 3

    if "several days" in duration_text:
        return 5

    if "week" in duration_text:
        return 7

    if "month" in duration_text:
        return 30

    return 1


def build_disease_input_dataframe(
    cleaned_text: str,
    age: Optional[int],
    gender: Optional[str],
    pain_score: Optional[int],
    body_part: Optional[str],
    symptoms: List[str],
    danger_terms: List[str],
    duration: Optional[str],
):
    """
    Must match disease training columns:

    user_text
    age_group
    sex
    body_part
    red_flags_present
    duration_days
    pain_score
    symptom_count
    """

    age_group = get_age_group(age)

    sex = str(gender).lower().strip() if gender else "not specified"

    selected_body_part = str(body_part).lower().strip() if body_part else "unknown"

    red_flags_present = "yes" if danger_terms else "no"

    selected_pain_score = pain_score if pain_score is not None else 5

    symptom_count = len(symptoms) if symptoms else 0

    duration_days = estimate_duration_days(duration)

    return pd.DataFrame([{
        "user_text": cleaned_text,
        "age_group": age_group,
        "sex": sex,
        "body_part": selected_body_part,
        "red_flags_present": red_flags_present,
        "duration_days": duration_days,
        "pain_score": selected_pain_score,
        "symptom_count": symptom_count
    }])


def predict_top_diseases(
    cleaned_text: str,
    age: Optional[int],
    gender: Optional[str],
    pain_score: Optional[int],
    body_part: Optional[str],
    symptoms: List[str],
    danger_terms: List[str],
    duration: Optional[str],
    top_n: int = 3
):
    """
    Supports:
    1. Full sklearn Pipeline trained with a DataFrame.
    2. Separate vectorizer + model trained on text only.

    Best case:
    model supports predict_proba(), so we return top 3 diseases.
    """

    model = get_disease_model()

    input_df = build_disease_input_dataframe(
        cleaned_text=cleaned_text,
        age=age,
        gender=gender,
        pain_score=pain_score,
        body_part=body_part,
        symptoms=symptoms,
        danger_terms=danger_terms,
        duration=duration
    )

    # Case 1: Full pipeline model
    if disease_vectorizer is None:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            top_indices = np.argsort(probabilities)[::-1][:top_n]

            return [
                {
                    "name": str(classes[index]),
                    "probability": round(float(probabilities[index]), 3)
                }
                for index in top_indices
            ]

        prediction = model.predict(input_df)[0]

        return [
            {
                "name": str(prediction),
                "probability": 1.0
            }
        ]

    # Case 2: Separate text vectorizer + model
    X = disease_vectorizer.transform([cleaned_text])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = model.classes_

        top_indices = np.argsort(probabilities)[::-1][:top_n]

        return [
            {
                "name": str(classes[index]),
                "probability": round(float(probabilities[index]), 3)
            }
            for index in top_indices
        ]

    prediction = model.predict(X)[0]

    return [
        {
            "name": str(prediction),
            "probability": 1.0
        }
    ]


# ============================================================
# API Routes
# ============================================================

@app.get("/")
def root():
    return {
        "message": "SACA Medical Triage API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "severity_model_loaded": severity_model is not None,
        "disease_model_loaded": disease_model is not None,
        "severity_model_path": str(SEVERITY_MODEL_PATH),
        "disease_model_path": str(DISEASE_MODEL_PATH),
        "severity_model_type": str(type(severity_model)),
        "disease_model_type": str(type(disease_model)),
        "severity_vectorizer_loaded": severity_vectorizer is not None,
        "disease_vectorizer_loaded": disease_vectorizer is not None,
        "severity_supports_predict": (
            hasattr(severity_model, "predict") if severity_model is not None else False
        ),
        "disease_supports_predict": (
            hasattr(disease_model, "predict") if disease_model is not None else False
        ),
        "disease_supports_predict_proba": (
            hasattr(disease_model, "predict_proba") if disease_model is not None else False
        )
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        print("\n🔹 NEW REQUEST 🔹")

        raw_text = payload.text or ""
        cleaned = normalize_text(raw_text)

        print("Cleaned text:", cleaned)

        symptoms = extract_symptoms(cleaned)
        symptoms = list(symptoms) if symptoms else []

        duration = extract_duration(cleaned)

        severity_words = extract_severity_words(cleaned)
        severity_words = list(severity_words) if severity_words else []

        danger_terms = detect_danger_terms(cleaned)
        danger_terms = list(danger_terms) if danger_terms else []

        negations = detect_negations(cleaned)
        negations = list(negations) if negations else []

        print("Symptoms:", symptoms)
        print("Duration:", duration)
        print("Severity words:", severity_words)
        print("Danger terms:", danger_terms)
        print("Negations:", negations)

        if cleaned.strip():
            severity, confidence = predict_severity(cleaned)
        else:
            severity, confidence = "mild", None

        severity = adjust_severity_with_rules(
            severity=severity,
            danger_terms=danger_terms,
            pain_score=payload.pain_score,
            body_part=payload.body_part
        )

        possible_diseases = predict_top_diseases(
            cleaned_text=cleaned,
            age=payload.age,
            gender=payload.gender,
            pain_score=payload.pain_score,
            body_part=payload.body_part,
            symptoms=symptoms,
            danger_terms=danger_terms,
            duration=duration,
            top_n=3
        )

        recommendation = generate_recommendation(
            severity=severity,
            danger_terms=danger_terms,
            pain_score=payload.pain_score,
            body_part=payload.body_part
        )

        print("Final severity:", severity)
        print("Possible diseases:", possible_diseases)

        return {
            "cleaned_text": cleaned,
            "severity": severity,
            "confidence": confidence,
            "possible_diseases": possible_diseases,
            "symptoms": symptoms,
            "duration": duration,
            "severity_words": severity_words,
            "danger_terms": danger_terms,
            "negations": negations,
            "recommendation": recommendation,
            "pain_score": payload.pain_score,
            "body_part": payload.body_part,
            "disclaimer": (
                "This is not a medical diagnosis. "
                "This system suggests possible conditions and severity guidance only. "
                "Please consult a healthcare professional."
            )
        }

    except Exception as e:
        print("\n❌ ERROR OCCURRED ❌")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))