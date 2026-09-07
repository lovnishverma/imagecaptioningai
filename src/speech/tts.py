import asyncio
import threading
import tempfile
from typing import Optional
import edge_tts
from src.config import CONFIG

# Voice mappings for supported locales
VOICE_MAP = {
    "English (US) - Aria": "en-US-AriaNeural",
    "English (US) - Guy": "en-US-GuyNeural",
    "English (UK) - Sonia": "en-GB-SoniaNeural",
    "Hindi - Swara": "hi-IN-SwaraNeural",
    "Hindi - Madhur": "hi-IN-MadhurNeural",
}

def init_tts_loop() -> asyncio.AbstractEventLoop:
    """Create a dedicated event loop for TTS in a background thread."""
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_run, daemon=True).start()
    return loop

_TTS_LOOP = init_tts_loop()

def text_to_speech(text: str, voice_name: str = "English (US) - Aria") -> Optional[str]:
    """Convert text to speech, returning the audio file path."""
    if not text or not text.strip():
        return None

    try:
        voice_id = VOICE_MAP.get(voice_name, "en-US-AriaNeural")

        async def _generate():
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{CONFIG.AUDIO_FORMAT}") as f:
                path = f.name
            communicate = edge_tts.Communicate(
                text.strip(),
                voice=voice_id,
                rate=CONFIG.TTS_RATE,
            )
            await communicate.save(path)
            return path

        future = asyncio.run_coroutine_threadsafe(_generate(), _TTS_LOOP)
        return future.result(timeout=CONFIG.TTS_TIMEOUT)
    except Exception as e:
        print(f"TTS error: {e}")
        return None
