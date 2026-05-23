import re
from typing import List, Optional


DANGER_TERMS = [
    "chest pain",
    "crushing chest pain",
    "difficulty breathing",
    "shortness of breath",
    "cannot breathe",
    "blue lips",
    "severe bleeding",
    "unconscious",
    "loss of consciousness",
    "fainting",
    "collapse",
    "seizure",
    "repeated seizure",
    "face drooping",
    "arm weakness",
    "speech difficulty",
    "cannot speak",
    "stroke",
    "heart attack",
    "vomiting blood",
    "blood in stool",
    "black stool",
    "severe abdominal pain",
    "severe dehydration",
    "confusion",
    "suicidal",
    "self harm"
]


NEGATION_PATTERNS = [
    r"\bno\s+\w+",
    r"\bnot\s+\w+",
    r"\bwithout\s+\w+",
    r"\bdenies\s+\w+",
    r"\bdo\s+not\s+have\s+\w+",
    r"\bdoes\s+not\s+have\s+\w+"
]


def detect_danger_terms(text: str) -> List[str]:
    text = text.lower()
    found = []

    for term in DANGER_TERMS:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text):
            found.append(term)

    return sorted(set(found))


def detect_negations(text: str) -> List[str]:
    text = text.lower()
    found = []

    for pattern in NEGATION_PATTERNS:
        matches = re.findall(pattern, text)
        found.extend(matches)

    return sorted(set(found))


def adjust_severity_with_rules(
    severity: str,
    danger_terms: List[str],
    pain_score: Optional[int] = None,
    body_part: Optional[str] = None
) -> str:
    severity = str(severity).lower().strip()
    body_part = str(body_part).lower().strip() if body_part else ""

    if danger_terms:
        return "severe"

    if pain_score is not None:
        if pain_score >= 9:
            return "severe"

        if pain_score >= 7 and severity == "mild":
            return "moderate"

    if body_part == "chest" and pain_score is not None and pain_score >= 7:
        return "severe"

    if body_part == "head" and pain_score is not None and pain_score >= 9:
        return "severe"

    if body_part in ["abdomen", "back"] and pain_score is not None and pain_score >= 8:
        if severity == "mild":
            return "moderate"

    return severity


def generate_recommendation(
    severity: str,
    danger_terms: List[str],
    pain_score: Optional[int] = None,
    body_part: Optional[str] = None
) -> str:
    severity = str(severity).lower().strip()
    body_part = str(body_part).lower().strip() if body_part else ""

    if danger_terms:
        return (
            "Urgent warning signs were detected. "
            "Seek immediate medical attention. If this is an emergency in Australia, call 000."
        )

    if body_part == "chest" and pain_score is not None and pain_score >= 7:
        return (
            "Chest pain with a high pain score was detected. "
            "Seek urgent medical attention immediately."
        )

    if body_part == "head" and pain_score is not None and pain_score >= 9:
        return (
            "Severe head pain was detected. "
            "Seek urgent medical attention immediately."
        )

    if severity == "mild":
        return (
            "Symptoms appear mild. Monitor your condition, rest if appropriate, "
            "and seek routine medical advice if symptoms continue or worsen."
        )

    if severity == "moderate":
        return (
            "Medical review is recommended soon. "
            "Please contact a GP, clinic, or community health worker."
        )

    if severity == "severe":
        return (
            "Symptoms may be serious. Seek urgent medical attention immediately. "
            "If this is an emergency in Australia, call 000."
        )

    return "Please consult a healthcare professional for advice."