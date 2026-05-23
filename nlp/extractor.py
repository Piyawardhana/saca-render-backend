import re
from typing import List, Optional


SYMPTOM_KEYWORDS = [
    "fever",
    "high fever",
    "cough",
    "dry cough",
    "persistent cough",
    "sore throat",
    "runny nose",
    "blocked nose",
    "headache",
    "migraine",
    "dizziness",
    "nausea",
    "vomiting",
    "diarrhea",
    "abdominal pain",
    "stomach pain",
    "chest pain",
    "chest tightness",
    "shortness of breath",
    "difficulty breathing",
    "cannot breathe",
    "wheezing",
    "fatigue",
    "tiredness",
    "body aches",
    "back pain",
    "rash",
    "itching",
    "swelling",
    "seizure",
    "confusion",
    "fainting",
    "loss of consciousness",
    "arm weakness",
    "face drooping",
    "speech difficulty",
    "blood in stool",
    "vomiting blood"
]


SEVERITY_WORDS = [
    "mild",
    "slight",
    "minor",
    "moderate",
    "bad",
    "worse",
    "worsening",
    "severe",
    "serious",
    "extreme",
    "unbearable",
    "crushing",
    "cannot",
    "constant",
    "persistent",
    "sudden",
    "rapidly"
]


DURATION_PATTERNS = [
    r"\bfor\s+\d+\s+day[s]?\b",
    r"\bfor\s+\d+\s+week[s]?\b",
    r"\bfor\s+\d+\s+month[s]?\b",
    r"\bsince\s+this\s+morning\b",
    r"\bsince\s+yesterday\b",
    r"\btoday\b",
    r"\byesterday\b",
    r"\bfor\s+a\s+few\s+days\b",
    r"\bfor\s+several\s+days\b",
    r"\bfor\s+a\s+week\b",
    r"\bright\s+now\b",
    r"\bsuddenly\b",
    r"\bgetting\s+worse\b",
    r"\bworsening\b"
]


def extract_symptoms(text: str) -> List[str]:
    text = text.lower()
    found = []

    for symptom in SYMPTOM_KEYWORDS:
        pattern = r"\b" + re.escape(symptom) + r"\b"
        if re.search(pattern, text):
            found.append(symptom)

    return sorted(set(found))


def extract_duration(text: str) -> Optional[str]:
    text = text.lower()
    found = []

    for pattern in DURATION_PATTERNS:
        matches = re.findall(pattern, text)
        found.extend(matches)

    if not found:
        return None

    return ", ".join(sorted(set(found)))


def extract_severity_words(text: str) -> List[str]:
    text = text.lower()
    found = []

    for word in SEVERITY_WORDS:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, text):
            found.append(word)

    return sorted(set(found))