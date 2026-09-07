def create_aria_live_region() -> str:
    """Returns HTML string for screen reader announcements."""
    return '<div class="sr-only" aria-live="assertive" id="aria-live-region" role="status"></div>'

def announce_status(message: str) -> str:
    """Formats a message to be read by screen readers immediately."""
    return f'<div id="sightline-status" role="status" aria-live="assertive">{message}</div>'
