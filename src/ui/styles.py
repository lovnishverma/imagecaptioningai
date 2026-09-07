CSS = """
/* Base Variables */
:root {
    --accent: #2563eb;
    --accent-hover: #1d4ed8;
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --text-primary: #1e293b;
    --border: #e2e8f0;
    --radius: 12px;
}

/* High Contrast Support */
body.hc {
    filter: contrast(1.7) brightness(1.05);
}

.gr-button {
    min-height: 80px !important; /* Larger touch targets */
    font-size: 1.3em !important;
    border-radius: var(--radius) !important;
    font-weight: bold !important;
    border: 4px solid transparent !important;
    transition: all 0.2s ease-in-out;
}

.gr-button:focus, .gr-button:hover {
    border: 4px solid #FFD700 !important; /* High contrast yellow focus ring */
    outline: none !important;
    transform: scale(1.02);
}

.gr-button-primary {
    background: #000000 !important;
    color: #FFFFFF !important;
    border: 4px solid #FFFFFF !important;
}

#sightline-status {
    background: #000000;
    color: #00FF00; /* High contrast terminal green */
    padding: 20px;
    border-radius: var(--radius);
    font-size: 1.4em;
    font-weight: 800;
    text-align: center;
    border: 3px solid #00FF00;
    box-shadow: 0 4px 6px rgba(0,255,0,0.2);
}

/* Screen reader only */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
}
"""
