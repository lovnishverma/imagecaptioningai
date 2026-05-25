"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ECHOLENS — Realtime Vision Assistant for Blind & Low-Vision Users         ║
║                                                                              ║
║  Keyboard:  D = Describe  ·  R = Toggle realtime  ·  Esc = Stop  ·  P = Repeat ║
║  Voice Commands: Click "Enable Voice Commands" for hands-free control       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import re
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import AutoModelForCausalLM, AutoProcessor

# ── Suppress noisy warnings ─────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", message=".*Using the model.*inference mode.*")

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════


class Config:
    """Central configuration — tweak values here."""

    # Timing
    CAPTURE_INTERVAL: float = 3.0  # seconds between realtime captures
    SCENE_THRESHOLD: float = 0.10  # dHash distance to treat as "same scene"
    DEBOUNCE_MS: int = 800  # ms to debounce rapid requests
    MAX_DIM: int = 768  # downscale before inference
    HASH_SIZE: int = 16  # perceptual hash grid size

    # Audio
    TTS_TIMEOUT: float = 12.0
    TTS_RATE: str = "+8%"  # slightly faster speech
    AUDIO_FORMAT: str = "mp3"
    MAX_QUEUE_SIZE: int = 3  # max pending audio announcements

    # Model
    MODEL_NAME: str = "microsoft/Florence-2-base"
    MAX_NEW_TOKENS: Dict[str, int] = field(default_factory=lambda: {
        "<CAPTION>": 64,
        "<DETAILED_CAPTION>": 120,
        "<MORE_DETAILED_CAPTION>": 200,
        "<OD>": 256,
        "<OCR>": 300,
    })

    # UI
    APP_NAME: str = "EchoLens"
    APP_VERSION: str = "2.0"


CONFIG = Config()

# ═══════════════════════════════════════════════════════════════
#  VOICE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

VOICE_MAP: Dict[str, str] = {
    "Aria — Female US": "en-US-AriaNeural",
    "Guy — Male US": "en-US-GuyNeural",
    "Jenny — Female US": "en-US-JennyNeural",
    "Sonia — Female UK": "en-GB-SoniaNeural",
    "Ryan — Male UK": "en-GB-RyanNeural",
    "Emily — Female Australia": "en-AU-EmilyNeural",
    "William — Male Australia": "en-AU-WilliamNeural",
    "Natasha — Female Australia": "en-AU-NatashaNeural",
}

TASKS: Dict[str, str] = {
    "Quick Caption": "<CAPTION>",
    "Describe Scene": "<DETAILED_CAPTION>",
    "Detailed Description": "<MORE_DETAILED_CAPTION>",
    "Read Text (OCR)": "<OCR>",
    "Detect Objects": "<OD>",
}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "Quick Caption": "A brief one-sentence description",
    "Describe Scene": "A paragraph describing the scene",
    "Detailed Description": "A thorough multi-sentence description",
    "Read Text (OCR)": "Reads any visible text aloud",
    "Detect Objects": "Names objects and their locations",
}


# ═══════════════════════════════════════════════════════════════
#  DEVICE & MODEL LOADING
# ═══════════════════════════════════════════════════════════════


def get_device() -> str:
    """Select best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE: str = get_device()
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"🖥️  Device: {DEVICE.upper()}")
print(f"🔢 Dtype: {DTYPE}")

# ── Model Loading ──────────────────────────────────────────────
_model_loaded = threading.Event()

processor: Optional[AutoProcessor] = None
model: Optional[AutoModelForCausalLM] = None


def _load_model():
    """Load Florence-2 model in background thread."""
    global model, processor
    try:
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG.MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=DTYPE,
        ).to(DEVICE).eval()

        processor = AutoProcessor.from_pretrained(
            CONFIG.MODEL_NAME,
            trust_remote_code=True,
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise


# Load synchronously on startup (can be made async if needed)
_load_model()
_model_loaded.set()

# ── Background Warmup ──────────────────────────────────────────
_warmup_done = threading.Event()


def _warmup_model():
    """Run a dummy inference to warm up CUDA kernels."""
    if model is None or processor is None:
        return
    try:
        dummy = Image.new("RGB", (224, 224), 128)
        inputs = processor(text="<CAPTION>", images=dummy, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=10,
                num_beams=1,
            )
        _warmup_done.set()
        print("🔥 Model warmed up")
    except Exception as e:
        print(f"Warmup warning: {e}")


threading.Thread(target=_warmup_model, daemon=True).start()


# ═══════════════════════════════════════════════════════════════
#  AUDIO QUEUE SYSTEM
# ═══════════════════════════════════════════════════════════════


class AudioQueue:
    """Thread-safe FIFO audio queue with interruption support."""

    def __init__(self, max_size: int = 3):
        self._queue: deque[Tuple[str, str]] = deque()  # (text, audio_path)
        self._current: Optional[str] = None
        self._lock = threading.Lock()
        self._counter = 0
        self._max_size = max_size

    def enqueue(self, text: str, audio_path: str) -> Optional[str]:
        """Add audio to queue. Returns the path to play (or None if queue full)."""
        with self._lock:
            if len(self._queue) >= self._max_size:
                # Remove oldest
                oldest = self._queue.popleft()
                self._safe_delete(oldest[1])
            self._queue.append((text, audio_path))
            self._counter += 1
            return audio_path

    def dequeue(self) -> Optional[Tuple[str, str]]:
        """Get next audio item."""
        with self._lock:
            if self._queue:
                item = self._queue.popleft()
                self._current = item[1]
                return item
            return None

    def clear(self):
        """Clear all queued audio and delete files."""
        with self._lock:
            for _, path in self._queue:
                self._safe_delete(path)
            self._queue.clear()
            self._current = None

    def interrupt(self):
        """Interrupt current and clear queue."""
        self.clear()

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    @staticmethod
    def _safe_delete(path: str):
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass


# Global audio queue
AUDIO_QUEUE = AudioQueue(max_size=CONFIG.MAX_QUEUE_SIZE)


# ═══════════════════════════════════════════════════════════════
#  TTS ENGINE (edge-tts)
# ═══════════════════════════════════════════════════════════════


def init_tts_loop() -> asyncio.AbstractEventLoop:
    """Create a dedicated event loop for TTS in a background thread."""
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_run, daemon=True).start()
    return loop


_TTS_LOOP = init_tts_loop()


def text_to_speech(text: str, voice_id: str = "en-US-AriaNeural") -> Optional[str]:
    """Convert text to speech, returning the audio file path."""
    if not text or not text.strip():
        return None

    try:
        import tempfile

        import edge_tts

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


# ═══════════════════════════════════════════════════════════════
#  APPLICATION STATE
# ═══════════════════════════════════════════════════════════════


@dataclass
class AppState:
    """Thread-safe application state."""

    # Scene hashing
    last_hash: Optional[bytes] = None
    last_task: str = ""
    last_text: str = ""
    last_audio: Optional[str] = None

    # Realtime
    realtime_active: bool = False
    last_capture_time: float = 0.0

    # History
    history: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 50

    # Stats
    total_describes: int = 0
    total_realtime_captures: int = 0

    # Lock
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _tmp_files: List[Tuple[str, float]] = field(default_factory=list)

    def update(self, hash_val: bytes, task: str, text: str, audio: Optional[str]):
        """Update state with new capture results."""
        with self._lock:
            self._cleanup_old_files()
            if self.last_audio and os.path.exists(self.last_audio):
                try:
                    os.unlink(self.last_audio)
                except OSError:
                    pass
            self.last_hash = hash_val
            self.last_task = task
            self.last_text = text
            self.last_audio = audio
            if audio:
                self._tmp_files.append((audio, time.time()))
            # Add to history
            self.history.insert(0, {
                "time": time.strftime("%H:%M:%S"),
                "task": task,
                "text": text,
            })
            if len(self.history) > self.max_history:
                self.history = self.history[: self.max_history]

    def is_duplicate(self, hash_val: bytes, task: str) -> bool:
        """Check if this hash+task combination was already processed."""
        with self._lock:
            return (
                self.last_hash is not None
                and self.last_hash == hash_val
                and self.last_task == task
                and self.last_text != ""
            )

    def get_last(self) -> Tuple[str, Optional[str]]:
        """Get last description text and audio."""
        with self._lock:
            return self.last_text, self.last_audio

    def add_stat(self, key: str):
        with self._lock:
            if key == "describe":
                self.total_describes += 1
            elif key == "realtime":
                self.total_realtime_captures += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "describes": self.total_describes,
                "realtime_captures": self.total_realtime_captures,
                "history_count": len(self.history),
            }

    def _cleanup_old_files(self):
        """Remove temp files older than 5 minutes."""
        now = time.time()
        keep = []
        for path, ts in self._tmp_files:
            if now - ts > 300:  # 5 minutes
                try:
                    os.unlink(path)
                except OSError:
                    pass
            else:
                keep.append((path, ts))
        self._tmp_files = keep


# Global state
APP_STATE = AppState()


# ═══════════════════════════════════════════════════════════════
#  IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════


def compute_hash(image: Image.Image, size: int = 16) -> bytes:
    """Compute difference hash (dHash) for scene change detection."""
    gray = image.resize((size + 1, size), Image.LANCZOS).convert("L")
    pixels = list(gray.getdata())
    return bytes(
        1 if pixels[y * (size + 1) + x] > pixels[y * (size + 1) + x + 1] else 0
        for y in range(size)
        for x in range(size)
    )


def hash_distance(a: Optional[bytes], b: Optional[bytes]) -> float:
    """Compute normalized Hamming distance between two hashes."""
    if a is None or b is None:
        return 1.0
    if len(a) != len(b):
        return 1.0
    return sum(x != y for x, y in zip(a, b)) / len(a)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Resize image for inference while preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= CONFIG.MAX_DIM:
        return image
    scale = CONFIG.MAX_DIM / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS)


def auto_enhance(image: Image.Image) -> Image.Image:
    """Auto-enhance image for better vision model performance."""
    # Slight contrast boost helps Florence-2 on low-light images
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    return image


# ═══════════════════════════════════════════════════════════════
#  CORE VISION INFERENCE
# ═══════════════════════════════════════════════════════════════


def run_inference(image: Image.Image, task_label: str) -> str:
    """Run Florence-2 inference on an image."""
    if model is None or processor is None:
        return "Error: Model not loaded. Please wait or restart."

    task_token = TASKS.get(task_label, "<CAPTION>")
    max_tokens = CONFIG.MAX_NEW_TOKENS.get(task_token, 64)

    try:
        # Preprocess
        image = preprocess_image(image)
        image = auto_enhance(image)

        # Prepare inputs
        inputs = processor(
            text=task_token,
            images=image,
            return_tensors="pt",
        ).to(DEVICE)

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )

        # Decode
        raw_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        result = processor.post_process_generation(
            raw_text,
            task=task_token,
            image_size=(image.width, image.height),
        )

        # Format output based on task
        if task_token == "<OD>":
            od_data = result.get("<OD>", {})
            return format_object_detection(od_data)
        elif task_token == "<OCR>":
            text_found = result.get("<OCR>", "").strip()
            if not text_found:
                return "No text detected in the image."
            return f"Text found: {text_found}"
        else:
            caption = result.get(task_token, "").strip()
            if not caption:
                return "I couldn't understand what's in the image. Please try again."
            return caption

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "The image is too large for memory. Try a smaller image."
    except Exception as e:
        print(f"Inference error: {e}")
        return f"Sorry, I had trouble analyzing that image. Please try again."


def format_object_detection(od_data: Dict) -> str:
    """Format object detection results into natural language."""
    if not od_data or not od_data.get("labels"):
        return "No objects detected in the image."

    labels = od_data.get("labels", [])
    bboxes = od_data.get("bboxes", [])

    if not labels:
        return "No objects detected in the image."

    # Build object list with positions
    objects: List[Tuple[str, str]] = []
    for label, bbox in zip(labels, bboxes):
        x1, _, x2, _ = bbox
        cx = (x1 + x2) / 2
        # Florence uses 0-999 coordinate space
        if cx < 333:
            pos = "on the left"
        elif cx < 666:
            pos = "in the center"
        else:
            pos = "on the right"
        objects.append((label.strip(), pos))

    # Deduplicate (keep first occurrence of each label type)
    seen: set = set()
    unique: List[Tuple[str, str]] = []
    for lbl, pos in objects:
        key = lbl.lower()
        if key and key not in seen:
            seen.add(key)
            unique.append((lbl, pos))

    if not unique:
        return "No objects detected in the image."

    # Format naturally
    if len(unique) == 1:
        lbl, pos = unique[0]
        return f"I see {lbl} {pos}."

    parts = [f"{lbl} {pos}" for lbl, pos in unique]

    if len(parts) <= 5:
        return "I see " + ", ".join(parts[:-1]) + f", and {parts[-1]}."
    else:
        summary = ", ".join(parts[:5])
        return f"I see {len(unique)} objects: {summary}, and {len(unique) - 5} more."


# ═══════════════════════════════════════════════════════════════
#  HANDLER FUNCTIONS
# ═══════════════════════════════════════════════════════════════


def describe_now(image, task_label: str, voice_name: str):
    """
    Manual describe handler.
    Streams words visually, then returns final text + audio.
    """
    if image is None:
        yield "📷 Please open the camera or upload an image first.", None, "Waiting for image..."
        return

    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Compute hash
    img_hash = compute_hash(image)
    task_key = TASKS.get(task_label, "<CAPTION>")

    # Check cache
    if APP_STATE.is_duplicate(img_hash, task_key):
        text, audio = APP_STATE.get_last()
        yield text, audio, f"✓ Cached result • {task_label}"
        return

    # Run inference
    yield "⏳ Analyzing image...", None, "Processing..."

    caption = run_inference(image, task_label)
    APP_STATE.add_stat("describe")

    # Stream words
    words = caption.split()
    partial = ""
    for i, w in enumerate(words):
        partial += (" " if partial else "") + w
        if (i + 1) % 3 == 0 or i == len(words) - 1:
            yield partial, None, f"⏳ Speaking... ({i + 1}/{len(words)} words)"

    # Generate TTS
    voice_id = VOICE_MAP.get(voice_name, "en-US-AriaNeural")
    audio_path = text_to_speech(caption, voice_id)

    # Update state
    APP_STATE.update(img_hash, task_key, caption, audio_path)

    yield caption, audio_path, f"✓ {task_label} • {len(words)} words"


def handle_upload(image, task_label: str, voice_name: str):
    """Handle uploaded image — same as describe."""
    yield from describe_now(image, task_label, voice_name)


def handle_realtime_stream(image, task_label: str, voice_name: str, rt_active: bool):
    """
    Called automatically by webcam.stream() every CAPTURE_INTERVAL seconds.
    Only processes if realtime toggle is ON.
    """
    if not rt_active:
        return gr.update(), gr.update(), "Realtime paused — press R to start"

    if image is None:
        return gr.update(), gr.update(), "No camera feed detected"

    # Convert to PIL
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Debounce check
    now = time.time()
    if now - APP_STATE.last_capture_time < 1.0:
        return gr.update(), gr.update(), "⏳ Debouncing..."
    APP_STATE.last_capture_time = now

    # Compute hash
    img_hash = compute_hash(image)
    task_key = TASKS.get(task_label, "<CAPTION>")

    # Scene change detection
    if APP_STATE.last_hash is not None:
        dist = hash_distance(img_hash, APP_STATE.last_hash)
        if dist < CONFIG.SCENE_THRESHOLD:
            return (
                gr.update(),
                gr.update(),
                f"🟢 Realtime active • Scene unchanged (similarity: {1 - dist:.0%})",
            )

    # Cache check
    if APP_STATE.is_duplicate(img_hash, task_key):
        text, audio = APP_STATE.get_last()
        return (
            text,
            audio,
            f"🟢 Realtime active • Used cached result",
        )

    # Run inference
    caption = run_inference(image, task_label)
    APP_STATE.add_stat("realtime")

    # Generate TTS
    voice_id = VOICE_MAP.get(voice_name, "en-US-AriaNeural")
    audio_path = text_to_speech(caption, voice_id)

    # Update state
    APP_STATE.update(img_hash, task_key, caption, audio_path)

    dist = hash_distance(img_hash, APP_STATE.last_hash) if APP_STATE.last_hash else 1.0
    status = f"🟢 Realtime active • Scene changed ({1 - dist:.0%} similar) • {len(caption.split())} words"

    return caption, audio_path, status


def toggle_realtime(current: bool) -> Tuple[bool, str, str]:
    """Toggle realtime mode on/off."""
    new_state = not current
    if new_state:
        label = "🟢 Stop Realtime (R)"
        status = "🟢 Realtime ON — describing every 3 seconds"
    else:
        AUDIO_QUEUE.interrupt()
        label = "⚫ Start Realtime (R)"
        status = "⚫ Realtime OFF — press R or click to start"
    return new_state, label, status


def repeat_last(voice_name: str):
    """Repeat the last description."""
    text, _ = APP_STATE.get_last()
    if not text:
        return "No previous description to repeat.", None, "No history available"

    voice_id = VOICE_MAP.get(voice_name, "en-US-AriaNeural")
    audio_path = text_to_speech(text, voice_id)

    return text, audio_path, "🔁 Repeated last description"


def stop_all():
    """Stop all audio and clear state."""
    AUDIO_QUEUE.interrupt()
    return "", None, "⏹ Stopped — press D to describe or R for realtime"


def get_history() -> str:
    """Get formatted history."""
    if not APP_STATE.history:
        return "No descriptions yet."
    lines = []
    for i, item in enumerate(APP_STATE.history[:10], 1):
        lines.append(f"{i}. [{item['time']}] {item['task']}: {item['text'][:80]}...")
    return "\n".join(lines)


def get_stats() -> str:
    """Get usage statistics."""
    stats = APP_STATE.get_stats()
    return (
        f"📊 Statistics:\n"
        f"• Manual describes: {stats['describes']}\n"
        f"• Realtime captures: {stats['realtime_captures']}\n"
        f"• History entries: {stats['history_count']}"
    )


# ═══════════════════════════════════════════════════════════════
#  CSS STYLES
# ═══════════════════════════════════════════════════════════════

CSS = """
/* ── Base ─────────────────────────────────────────────────── */
:root {
    --accent: #2563eb;
    --accent-hover: #1d4ed8;
    --success: #059669;
    --warning: #d97706;
    --danger: #dc2626;
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-dark: #0f172a;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border: #e2e8f0;
    --radius: 12px;
    --shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-lg: 0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1);
}

/* ── Font size modes ──────────────────────────────────────── */
body.fs-normal  { --base-size: 16px; }
body.fs-large   { --base-size: 20px; }
body.fs-xlarge  { --base-size: 26px; }

body {
    font-size: var(--base-size, 16px) !important;
}

/* ── High Contrast Mode ───────────────────────────────────── */
body.hc {
    filter: contrast(1.7) brightness(1.05);
}
body.hc .gr-button {
    border: 2px solid #000 !important;
}
body.hc .gr-input,
body.hc .gr-textbox textarea {
    border: 2px solid #000 !important;
}

/* ── Layout ───────────────────────────────────────────────── */
.gr-button {
    min-height: 52px !important;
    font-size: var(--base-size, 16px) !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
    box-shadow: var(--shadow) !important;
}
.gr-button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-lg) !important;
}
.gr-button:active {
    transform: translateY(0);
}

/* Primary button */
.gr-button-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent-hover)) !important;
    border: none !important;
}

/* ── Textbox ──────────────────────────────────────────────── */
.gr-textbox textarea {
    font-size: calc(var(--base-size, 16px) * 1.15) !important;
    line-height: 1.7 !important;
    border-radius: var(--radius) !important;
    padding: 14px !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Status Bar ───────────────────────────────────────────── */
#echo-status {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: #f1f5f9;
    padding: 14px 20px;
    border-radius: var(--radius);
    font-size: calc(var(--base-size, 16px) * 0.95);
    font-weight: 600;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
    border-left: 4px solid var(--accent);
    transition: all 0.3s ease;
}

/* ── Cards ────────────────────────────────────────────────── */
.echo-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
}

/* ── Keyboard Shortcuts Display ───────────────────────────── */
.echo-kbd {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: #e2e8f0;
    border-radius: 6px;
    font-size: calc(var(--base-size, 16px) * 0.8);
    font-family: monospace;
    font-weight: 600;
    color: #334155;
}

/* ── Tips Box ─────────────────────────────────────────────── */
.echo-tips {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #a7f3d0;
    border-radius: var(--radius);
    padding: 18px;
    margin-top: 14px;
    font-size: calc(var(--base-size, 16px) * 0.9);
    line-height: 1.7;
}

/* ── Radio buttons ────────────────────────────────────────── */
.gr-radio {
    font-size: calc(var(--base-size, 16px) * 0.95) !important;
}

/* ── Dropdown ─────────────────────────────────────────────── */
.gr-dropdown {
    font-size: calc(var(--base-size, 16px) * 0.95) !important;
}

/* ── Section headers ──────────────────────────────────────── */
.echo-section-title {
    font-size: calc(var(--base-size, 16px) * 1.2);
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
}

/* ── Accessibility Toolbar ───────────────────────────────── */
.echo-toolbar {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 16px;
    padding: 12px;
    background: var(--bg-secondary);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    align-items: center;
}

.echo-toolbar button {
    padding: 8px 16px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: white;
    cursor: pointer;
    font-weight: 600;
    font-size: calc(var(--base-size, 16px) * 0.85);
    transition: all 0.15s;
}
.echo-toolbar button:hover {
    background: #e2e8f0;
    transform: translateY(-1px);
}

/* ── Stats display ────────────────────────────────────────── */
.echo-stats {
    font-family: monospace;
    font-size: calc(var(--base-size, 16px) * 0.85);
    color: var(--text-secondary);
    background: var(--bg-secondary);
    padding: 10px 14px;
    border-radius: var(--radius);
    margin-top: 10px;
}

/* ── Responsive ───────────────────────────────────────────── */
@media (max-width: 768px) {
    .gr-button {
        width: 100% !important;
        min-height: 56px !important;
    }
    .echo-toolbar {
        flex-direction: column;
        align-items: stretch;
    }
    .echo-toolbar button {
        width: 100%;
    }
}

/* ── Focus indicators for accessibility ───────────────────── */
button:focus-visible,
.gr-button:focus-visible {
    outline: 3px solid var(--accent) !important;
    outline-offset: 2px !important;
}

/* ── Screen reader only ───────────────────────────────────── */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
}

/* ── Loading animation ────────────────────────────────────── */
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.echo-loading::after {
    content: "...";
    animation: pulse-dot 1.5s infinite;
}
"""

# ═══════════════════════════════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════════════════════════════


def build_ui() -> gr.Blocks:
    """Build the Gradio user interface."""

    with gr.Blocks(
        title=f"{CONFIG.APP_NAME} v{CONFIG.APP_VERSION} — Vision Assistant",
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            spacing_size="md",
            radius_size="md",
        ),
        analytics_enabled=False,
    ) as demo:

        # ── Live region for screen readers ───────────────────
        gr.HTML(
            '<div class="sr-only" aria-live="assertive" aria-atomic="true" '
            'id="aria-live-region" role="status"></div>'
        )

        # ── Status Bar ───────────────────────────────────────
        status_bar = gr.HTML(
            '<div id="echo-status" role="status" aria-live="polite">'
            "✅ Ready — Press D to describe what the camera sees"
            "</div>"
        )

        # ── Header ───────────────────────────────────────────
        gr.Markdown(
            f"# 👁️ {CONFIG.APP_NAME} — Realtime Vision Assistant",
            elem_classes=["echo-section-title"],
        )
        gr.Markdown(
            "Helping blind and visually impaired users understand their surroundings. "
            "Press **D** to describe, **R** for realtime mode, **P** to repeat."
        )

        # ── Accessibility Toolbar ────────────────────────────
        gr.HTML("""
        <div class="echo-toolbar" role="toolbar" aria-label="Accessibility controls">
            <span style="font-weight:600;color:#475569;font-size:0.9em">Text Size:</span>
            <button onclick="document.body.classList.remove('fs-large','fs-xlarge');document.body.classList.add('fs-normal')"
                aria-label="Normal text size">A</button>
            <button onclick="document.body.classList.remove('fs-large','fs-xlarge');document.body.classList.add('fs-large')"
                aria-label="Large text size" style="font-size:1.15em">A+</button>
            <button onclick="document.body.classList.remove('fs-large','fs-xlarge');document.body.classList.add('fs-xlarge')"
                aria-label="Extra large text size" style="font-size:1.3em">A++</button>
            <button onclick="document.body.classList.toggle('hc')"
                aria-label="Toggle high contrast mode"
                style="background:#1e293b;color:white">⬛ High Contrast</button>
            <span style="margin-left:auto;font-size:0.85em;color:#6b7280;align-self:center">
                <span class="echo-kbd">D</span> describe ·
                <span class="echo-kbd">R</span> realtime ·
                <span class="echo-kbd">P</span> repeat ·
                <span class="echo-kbd">Esc</span> stop
            </span>
        </div>
        """
        )

        # ── Realtime state (single source of truth) ──────────
        rt_state = gr.State(False)

        with gr.Row():
            # ════════════════════════════════════════════════
            #  LEFT COLUMN — Inputs
            # ════════════════════════════════════════════════
            with gr.Column(scale=1):

                # ── Camera ─────────────────────────────────
                webcam = gr.Image(
                    label="📷 Camera Feed",
                    type="numpy",
                    sources=["webcam"],
                    streaming=True,
                    height=260,
                    elem_id="echo-webcam",
                )

                # ── Upload ─────────────────────────────────
                upload = gr.Image(
                    label="📁 Or Upload Image",
                    type="numpy",
                    sources=["upload"],
                    height=140,
                    elem_id="echo-upload",
                )

                # ── Task Selection ─────────────────────────
                task_radio = gr.Radio(
                    choices=list(TASKS.keys()),
                    value="Quick Caption",
                    label="What should I do?",
                    info="Select the type of description you want",
                )

                # Task description
                task_info = gr.Textbox(
                    value=TASK_DESCRIPTIONS["Quick Caption"],
                    label="",
                    interactive=False,
                    max_lines=1,
                    show_label=False,
                    container=False,
                    elem_classes=["echo-stats"],
                )

                # ── Voice Selection ────────────────────────
                voice_dropdown = gr.Dropdown(
                    choices=list(VOICE_MAP.keys()),
                    value="Aria — Female US",
                    label="🔊 Voice",
                    info="Choose a voice for spoken descriptions",
                )

                # ── Describe Button ────────────────────────
                describe_btn = gr.Button(
                    "🔍 Describe Now (D)",
                    variant="primary",
                    size="lg",
                    elem_id="echo-describe-btn",
                )

                # ── Realtime Toggle ────────────────────────
                realtime_btn = gr.Button(
                    "⚫ Start Realtime (R)",
                    variant="secondary",
                    size="lg",
                    elem_id="echo-rt-btn",
                )

            # ════════════════════════════════════════════════
            #  RIGHT COLUMN — Output
            # ════════════════════════════════════════════════
            with gr.Column(scale=1):

                # ── Caption Output ─────────────────────────
                caption_box = gr.Textbox(
                    label="📝 Description",
                    lines=6,
                    interactive=False,
                    show_copy_button=True,
                    placeholder="Description will appear here...",
                    elem_id="echo-caption",
                )

                # ── Audio Output ───────────────────────────
                audio_player = gr.Audio(
                    label="🔊 Audio",
                    type="filepath",
                    autoplay=True,
                    elem_id="echo-audio",
                )

                # ── Action Buttons ─────────────────────────
                with gr.Row():
                    repeat_btn = gr.Button(
                        "🔁 Repeat Last (P)",
                        variant="secondary",
                        size="lg",
                        elem_id="echo-repeat-btn",
                    )
                    stop_btn = gr.Button(
                        "⏹ Stop All (Esc)",
                        variant="stop",
                        size="lg",
                        elem_id="echo-stop-btn",
                    )

                # ── Tips ───────────────────────────────────
                gr.HTML("""
                <div class="echo-tips" role="complementary" aria-label="Tips for users">
                    <strong style="color:#065f46;font-size:1.05em">💡 Tips:</strong><br>
                    • <strong>D</strong> — Describe what the camera sees right now<br>
                    • <strong>R</strong> — Start/stop auto-description every 3 seconds<br>
                    • <strong>P</strong> — Repeat the last description<br>
                    • <strong>Esc</strong> — Stop all audio and realtime mode<br>
                    • <strong>Read Text</strong> — Reads signs, labels, screens (OCR)<br>
                    • <strong>Detect Objects</strong> — Hear what's where in the scene
                </div>
                """)

        # ═══════════════════════════════════════════════════
        #  BOTTOM SECTION — Stats & History
        # ═══════════════════════════════════════════════════
        with gr.Accordion("📊 Session Statistics", open=False):
            stats_box = gr.Textbox(
                value="Press 'Get Stats' to see usage statistics",
                label="Statistics",
                interactive=False,
                lines=4,
            )
            stats_btn = gr.Button("Refresh Statistics", size="sm")

        # ═══════════════════════════════════════════════════
        #  EVENT WIRING
        # ═══════════════════════════════════════════════════

        # Update task description when task changes
        def update_task_info(task_label):
            return TASK_DESCRIPTIONS.get(task_label, "")

        task_radio.change(
            update_task_info,
            inputs=[task_radio],
            outputs=[task_info],
        )

        # Manual Describe
        describe_btn.click(
            describe_now,
            inputs=[webcam, task_radio, voice_dropdown],
            outputs=[caption_box, audio_player, status_bar],
            show_progress="minimal",
        )

        # Upload
        upload.change(
            handle_upload,
            inputs=[upload, task_radio, voice_dropdown],
            outputs=[caption_box, audio_player, status_bar],
            show_progress="minimal",
        )

        # Realtime Toggle
        realtime_btn.click(
            toggle_realtime,
            inputs=[rt_state],
            outputs=[rt_state, realtime_btn, status_bar],
        )

        # Realtime Stream
        webcam.stream(
            handle_realtime_stream,
            inputs=[webcam, task_radio, voice_dropdown, rt_state],
            outputs=[caption_box, audio_player, status_bar],
            stream_every=CONFIG.CAPTURE_INTERVAL,
            time_limit=None,
        )

        # Repeat
        repeat_btn.click(
            repeat_last,
            inputs=[voice_dropdown],
            outputs=[caption_box, audio_player, status_bar],
            show_progress=False,
        )

        # Stop
        stop_btn.click(
            stop_all,
            inputs=[],
            outputs=[caption_box, audio_player, status_bar],
            show_progress=False,
        )

        # Stats
        stats_btn.click(
            get_stats,
            inputs=[],
            outputs=[stats_box],
        )

        # ═══════════════════════════════════════════════════
        #  KEYBOARD SHORTCUTS (JavaScript)
        # ═══════════════════════════════════════════════════
        gr.HTML("""
        <script>
        document.addEventListener('keydown', function(e) {
            // Don't trigger shortcuts when typing in inputs
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
                return;
            }

            const key = e.key.toLowerCase();

            if (key === 'd') {
                e.preventDefault();
                const btn = document.getElementById('echo-describe-btn');
                if (btn) btn.click();
            }
            else if (key === 'r') {
                e.preventDefault();
                const btn = document.getElementById('echo-rt-btn');
                if (btn) btn.click();
            }
            else if (key === 'p') {
                e.preventDefault();
                const btn = document.getElementById('echo-repeat-btn');
                if (btn) btn.click();
            }
            else if (key === 'escape') {
                e.preventDefault();
                const btn = document.getElementById('echo-stop-btn');
                if (btn) btn.click();
            }
        });

        // Announce to screen readers
        function announce(message) {
            const live = document.getElementById('aria-live-region');
            if (live) {
                live.textContent = message;
                setTimeout(() => { live.textContent = ''; }, 1000);
            }
        }

        // Hook button clicks for announcements
        document.addEventListener('click', function(e) {
            const btn = e.target.closest('button');
            if (!btn) return;
            if (btn.id === 'echo-describe-btn') announce('Describing scene');
            if (btn.id === 'echo-rt-btn') announce('Toggling realtime mode');
            if (btn.id === 'echo-repeat-btn') announce('Repeating last description');
            if (btn.id === 'echo-stop-btn') announce('Stopping all audio');
        });
        </script>
        """)

    return demo


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"🚀 Starting {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}")
    print(f"   Device: {DEVICE.upper()}")
    print(f"   Tasks: {list(TASKS.keys())}")
    print(f"   Voices: {list(VOICE_MAP.keys())}")
    print(f"   Realtime interval: {CONFIG.CAPTURE_INTERVAL}s")

    demo = build_ui()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        favicon_path=None,
    )
