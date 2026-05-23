import re


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text).lower().strip()

    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "i'm": "i am",
        "i've": "i have",
        "i'd": "i would",
        "it's": "it is",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not"
    }

    for short, full in contractions.items():
        text = text.replace(short, full)

    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s/.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text