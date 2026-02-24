import re

def normalize_for_embedding(text: str) -> str:
    text = text.lower()
    text = re.sub(
        r"\b(who|what|is|are|the|best|top|find|show|me)\b",
        "",
        text
    )
    return text.strip()
