"""
Loads the trained model package and exposes a predict() function.
The model file is created by train_model.py (models/medivoice_triage_model.joblib).
"""
from pathlib import Path
import joblib

# project_root/app/triage_service.py -> project_root/models/...
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "medivoice_triage_model.joblib"

# Safety net: phrases that should always be treated as HIGH, regardless of the ML output.
RED_FLAGS = [
    "chest pain", "crushing chest", "cannot breathe", "can't breathe",
    "difficulty breathing", "severe bleeding", "unconscious", "fainted",
    "seizure", "cannot speak", "can't speak", "slurred speech",
    "face drooping", "arm feels weak", "suicidal", "overdose",
    "coughing blood", "vomiting blood",
]

RECOMMENDATIONS = {
    "low": "Symptoms appear mild. Rest, stay hydrated and monitor. Seek care if they worsen.",
    "moderate": "Please see a healthcare professional within 24-48 hours.",
    "high": "Seek emergency medical care immediately or call your local emergency number.",
}

DISCLAIMER = (
    "MediVoice AI provides preliminary triage guidance only and is not a "
    "medical diagnosis. Consult a qualified healthcare professional."
)

_package = None


def load_model():
    """Load once at app startup."""
    global _package
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    _package = joblib.load(MODEL_PATH)  # dict saved by train_model.py
    return _package


def predict(text: str, chief_complaint: str = "") -> dict:
    if _package is None:
        load_model()

    model = _package["model"]

    # Must match training: chief_complaint + " " + text
    input_text = f"{chief_complaint or ''} {text}".strip()

    probs = model.predict_proba([input_text])[0]
    classes = list(model.named_steps["classifier"].classes_)
    probabilities = {c: float(p) for c, p in zip(classes, probs)}

    level = max(probabilities, key=probabilities.get)
    confidence = probabilities[level]

    # Safety rules
    override = False
    lowered = input_text.lower()
    if any(flag in lowered for flag in RED_FLAGS) and level != "high":
        level, override = "high", True
    elif level == "low" and confidence < 0.55:
        # Uncertain "low" -> be cautious
        level, override = "moderate", True

    return {
        "triage_level": level,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "red_flag_override": override,
        "recommendation": RECOMMENDATIONS[level],
        "disclaimer": DISCLAIMER,
    }
