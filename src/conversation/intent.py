import re

def detect_intent(text: str) -> str:
    """Map natural language to a vision task token."""
    text = text.lower().strip()
    
    if re.search(r"read|sign|text|document|label|menu", text):
        return "<OCR>"
    elif re.search(r"where|object|nearby|around me|front of me", text):
        return "<OD>"
    elif re.search(r"detail|describe.*room", text):
        return "<MORE_DETAILED_CAPTION>"
    else:
        # Default to standard caption
        return "<DETAILED_CAPTION>"
