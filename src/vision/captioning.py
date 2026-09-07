def format_caption(raw_caption: str) -> str:
    """Format raw scene descriptions for speech clarity."""
    caption = raw_caption.strip()
    if not caption:
        return "I couldn't understand what's in the image."
    
    # Capitalize first letter, ensure period at the end
    if not caption.endswith('.'):
        caption += '.'
    caption = caption[0].upper() + caption[1:]
    
    return caption
