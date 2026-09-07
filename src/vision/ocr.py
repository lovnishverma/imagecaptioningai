def format_ocr(raw_text: str) -> str:
    """Format OCR text for speech clarity."""
    text_found = raw_text.strip()
    if not text_found:
        return "No text detected."
    
    # Add conversational framing
    return f"The text says: {text_found}"
