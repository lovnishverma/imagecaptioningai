"""
EchoLens — Realtime Vision Assistant for Blind & Low-Vision Users
=================================================================
Keyboard:  D = Describe now  ·  R = Toggle realtime  ·  Esc = Stop  ·  P = Repeat
"""

import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import edge_tts
import tempfile
import asyncio
import threading
import time
import os
import re
from io import BytesIO
from collections import Counter

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
CAPTURE_INTERVAL = 3.5         # seconds between realtime captures
HASH_THRESHOLD   = 0.12        # dHash distance to treat as "same scene"
MAX_DIM          = 768         # downscale before inference

VOICE_MAP = {
    "Aria (Female, US)":   "en-US-AriaNeural",
    "Guy (Male, US)":      "en-US-GuyNeural",
    "Jenny (Female, US)":  "en-US-JennyNeural",
    "Sonia (Female, UK)":  "en-GB-SoniaNeural",
    "Ryan (Male, UK)":     "en-GB-RyanNeural",
}

TASKS = {
    "Quick Caption":       "<CAPTION>",
    "Describe Scene":      "<MORE_DETAILED_CAPTION>",
    "Read Text (OCR)":     "<OCR>",
    "Detect Objects":      "<OD>",
}

MAX_TOKENS = {
    "<CAPTION>":                64,
    "<MORE_DETAILED_CAPTION>": 160,
    "<OD>":                    256,
    "<OCR>":                   300,
}

# ═══════════════════════════════════════════════════════════════
#  DEVICE & MODEL
# ═══════════════════════════════════════════════════════════════
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device).eval()

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True,
)

# ── Background warmup ────────────────────────────────────────
_warmup_done = threading.Event()

def _warmup():
    dummy = Image.new("RGB", (224, 224), 128)
    inp = processor(text="<CAPTION>", images=dummy, return_tensors="pt").to(device)
    with torch.inference_mode():
        model.generate(input_ids=inp["input_ids"],
                       pixel_values=inp["pixel_values"],
                       max_new_tokens=10, num_beams=1)
    _warmup_done.set()
    print("Model warmed up!")

threading.Thread(target=_warmup, daemon=True).start()

# ═══════════════════════════════════════════════════════════════
#  DEDICATED TTS EVENT LOOP
# ═══════════════════════════════════════════════════════════════
_tts_loop = asyncio.new_event_loop()

def _run_tts_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=_run_tts_loop, args=(_tts_loop,), daemon=True).start()

def text_to_speech(text: str, voice_id: str = "en-US-AriaNeural") -> str | None:
    if not text or not text.strip():
        return None
    try:
        async def _gen():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                path = f.name
            comm = edge_tts.Communicate(text.strip(), voice=voice_id, rate="+5%")
            await comm.save(path)
            return path
        future = asyncio.run_coroutine_threadsafe(_gen(), _tts_loop)
        return future.result(timeout=15)
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════
class State:
    def __init__(self):
        self.hash: bytes | None = None
        self.task: str = ""
        self.text: str = ""
        self.audio: str | None = None
        self.lock = threading.Lock()
        self._tmp_files: list[tuple[str, float]] = []

    def set(self, h: bytes, task: str, text: str, audio: str | None):
        with self.lock:
            self._cleanup()
            if self.audio and os.path.exists(self.audio):
                try: os.unlink(self.audio)
                except OSError: pass
            self.hash  = h
            self.task  = task
            self.text  = text
            self.audio = audio
            if audio:
                self._tmp_files.append((audio, time.time()))

    def _cleanup(self):
        now = time.time()
        keep = []
        for path, ts in self._tmp_files:
            if now - ts > 300:
                try: os.unlink(path)
                except OSError: pass
            else:
                keep.append((path, ts))
        self._tmp_files = keep

    def matches(self, h: bytes, task: str) -> bool:
        return (self.hash is not None
                and self.hash == h
                and self.task == task
                and self.text)

state = State()

# ═══════════════════════════════════════════════════════════════
#  IMAGE HELPERS
# ═══════════════════════════════════════════════════════════════

def dhash(img: Image.Image, size: int = 16) -> bytes:
    gray = img.resize((size + 1, size)).convert("L")
    px = list(gray.getdata())
    return bytes(
        1 if px[y * (size + 1) + x] > px[y * (size + 1) + x + 1] else 0
        for y in range(size) for x in range(size)
    )

def hash_dist(a: bytes | None, b: bytes | None) -> float:
    if a is None or b is None:
        return 1.0
    return sum(x != y for x, y in zip(a, b)) / max(len(a), 1)

def resize_for_inference(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_DIM:
        return img
    scale = MAX_DIM / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

# ═══════════════════════════════════════════════════════════════
#  CORE INFERENCE
# ═══════════════════════════════════════════════════════════════

def run_inference(image: Image.Image, task_label: str) -> str:
    task = TASKS.get(task_label, "<CAPTION>")
    image = resize_for_inference(image)
    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=MAX_TOKENS.get(task, 64),
            do_sample=False,
            num_beams=1,
        )
    raw = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        raw, task=task, image_size=(image.width, image.height)
    )
    if task == "<OD>":
        od_result = result.get("<OD>", {})          # ← this is a dict, not a string
        return _format_od(od_result)
    elif task == "<OCR>":
        text_found = result.get("<OCR>", "").strip()
        return f"Text found: {text_found}" if text_found else "No text detected."
    else:
        return result.get(task, "").strip()


def _format_od(od: dict) -> str:
    """od = {"bboxes": [[x1,y1,x2,y2], ...], "labels": ["cat", "dog", ...]}"""
    if not od or not od.get("labels"):
        return "No objects detected."

    labels = od.get("labels", [])
    bboxes = od.get("bboxes", [])

    objects: list[tuple[str, str]] = []
    for label, bbox in zip(labels, bboxes):
        x1, _, x2, _ = bbox
        cx = (x1 + x2) / 2
        # Florence-2 bboxes are in absolute pixels relative to image_size
        # Use 1/3 and 2/3 of image width as thresholds — but we don't have
        # image width here, so use the Florence coordinate space (0–999)
        pos = "on the left" if cx < 333 else ("in the center" if cx < 666 else "on the right")
        objects.append((label.strip(), pos))

    if not objects:
        return "No objects detected."

    # Deduplicate by label
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for lbl, pos in objects:
        key = lbl.lower()
        if key and key not in seen:
            seen.add(key)
            unique.append((lbl, pos))

    if len(unique) == 1:
        lbl, pos = unique[0]
        return f"I see {lbl} {pos}."
    parts = [f"{lbl} {pos}".strip() for lbl, pos in unique]
    if len(parts) <= 6:
        return "I see " + ", ".join(parts[:-1]) + f", and {parts[-1]}."
    return f"I see {len(unique)} objects: " + ", ".join(parts[:5]) + f", and {len(unique) - 5} more."

# ═══════════════════════════════════════════════════════════════
#  HANDLERS
# ═══════════════════════════════════════════════════════════════

def handle_describe(image, task_label: str, voice_name: str):
    """Manual describe — streams words visually, then returns audio."""
    if image is None:
        yield "Please open the camera or upload an image.", None
        return
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    h = dhash(image)
    task_key = TASKS.get(task_label, "<CAPTION>")

    with state.lock:
        if state.matches(h, task_key):
            yield state.text, state.audio
            return

    caption = run_inference(image, task_label)

    words = caption.split()
    partial = ""
    for i, w in enumerate(words):
        partial += (" " if partial else "") + w
        if (i + 1) % 4 == 0 or i == len(words) - 1:
            yield partial, None

    voice_id = VOICE_MAP.get(voice_name, "en-US-AriaNeural")
    audio = text_to_speech(caption, voice_id)
    state.set(h, task_key, caption, audio)
    yield caption, audio


def handle_realtime_stream(image, task_label: str, voice_name: str, rt_active: bool):
    """
    Called automatically by webcam.stream() every CAPTURE_INTERVAL seconds.
    Only processes if realtime toggle is ON.
    """
    if not rt_active:
        return gr.update(), gr.update()
    if image is None:
        return gr.update(), gr.update()

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    h = dhash(image)
    task_key = TASKS.get(task_label, "<CAPTION>")

    # Scene-change gate
    with state.lock:
        if state.hash is not None and hash_dist(h, state.hash) < HASH_THRESHOLD:
            return gr.update(), gr.update()

    # Cache check
    with state.lock:
        if state.matches(h, task_key):
            return state.text, state.audio

    caption = run_inference(image, task_label)
    voice_id = VOICE_MAP.get(voice_name, "en-US-AriaNeural")
    audio = text_to_speech(caption, voice_id)
    state.set(h, task_key, caption, audio)
    return caption, audio


def handle_upload(image, task_label: str, voice_name: str):
    yield from handle_describe(image, task_label, voice_name)


def handle_repeat(voice_name: str):
    with state.lock:
        text = state.text
    if not text:
        return gr.update(), gr.update(value=None)
    voice_id = VOICE_MAP.get(voice_name, "en-US-AriaNeural")
    audio = text_to_speech(text, voice_id)
    return text, audio


def handle_stop():
    return gr.update(value=""), gr.update(value=None)


def toggle_realtime(current: bool):
    """Flip the realtime state, return new state + updated button label."""
    new_state = not current
    if new_state:
        label = "🟢 Realtime ON — click to stop (R)"
        status = "Realtime ON — describing every 3.5 seconds"
    else:
        label = "⚫ Toggle Realtime (R)"
        status = "Realtime stopped."
    return new_state, label, status


# ═══════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════
CSS = """
body { font-size: 18px !important; }
.gr-button {
    min-height: 52px !important;
    font-size: 17px !important;
    border-radius: 12px !important;
    cursor: pointer;
}
.gr-textbox textarea {
    font-size: 20px !important;
    line-height: 1.7 !important;
}
body.hc { filter: contrast(1.6) brightness(1.1); }
body.fs-large  .gr-textbox textarea { font-size: 26px !important; }
body.fs-xlarge .gr-textbox textarea { font-size: 32px !important; }
#echo-status {
    background: #1e293b; color: #f8fafc;
    padding: 12px 18px; border-radius: 10px;
    font-size: 17px; margin-bottom: 12px;
    min-height: 44px; font-weight: 600;
}
@media (max-width: 768px) {
    .gr-button { width: 100% !important; min-height: 58px !important; }
}
"""

# ═══════════════════════════════════════════════════════════════
#  BUILD UI
# ═══════════════════════════════════════════════════════════════
with gr.Blocks(title="EchoLens — Vision Assistant for the Blind", css=CSS,
               theme=gr.themes.Soft()) as demo:

    # Realtime toggle state — single source of truth
    rt_active = gr.State(False)

    gr.HTML('<div id="echo-live" aria-live="assertive" aria-atomic="true" '
            'style="position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden" '
            'role="status"></div>')

    gr.HTML("""
    <div id="echo-status" role="status" aria-live="polite">
        Loading model, please wait...
    </div>
    """)

    gr.Markdown("# 👁️ EchoLens — Vision Assistant")
    gr.Markdown("Helping blind and visually impaired users understand their surroundings.")

    # Accessibility toolbar
    gr.HTML("""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">
        <button onclick="document.body.classList.remove('fs-large','fs-xlarge')"
            style="padding:8px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer;font-size:14px">A</button>
        <button onclick="document.body.classList.remove('fs-large','fs-xlarge');document.body.classList.add('fs-large')"
            style="padding:8px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer;font-size:17px">A+</button>
        <button onclick="document.body.classList.remove('fs-large','fs-xlarge');document.body.classList.add('fs-xlarge')"
            style="padding:8px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer;font-size:20px">A++</button>
        <button onclick="document.body.classList.toggle('hc')"
            style="padding:8px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer;font-size:14px;background:#1e293b;color:white">⬛ High Contrast</button>
        <span style="margin-left:auto;font-size:13px;color:#6b7280;align-self:center">
            <kbd>D</kbd> describe · <kbd>R</kbd> realtime · <kbd>P</kbd> repeat · <kbd>Esc</kbd> stop
        </span>
    </div>
    """)

    with gr.Row():
        # ── LEFT: camera ──────────────────────────────────────
        with gr.Column(scale=1):
            webcam_input = gr.Image(
                label="Camera", type="numpy", sources=["webcam"],
                elem_id="echo-webcam", height=224, streaming=True,
            )
            upload_input = gr.Image(
                label="Upload Image", type="numpy", sources=["upload"],
                elem_id="echo-upload", height=160,
            )
            task_choice = gr.Radio(
                choices=list(TASKS.keys()),
                value="Quick Caption",
                label="What should I do?",
            )
            voice_choice = gr.Dropdown(
                choices=list(VOICE_MAP.keys()),
                value="Aria (Female, US)",
                label="Voice",
            )

            describe_btn = gr.Button(
                "🔍 Describe Now (D)", variant="primary", size="lg",
                elem_id="echo-describe-btn",
            )

            # ── Realtime toggle — pure Gradio button + State ──
            realtime_btn = gr.Button(
                "⚫ Toggle Realtime (R)",
                variant="secondary", size="lg",
                elem_id="echo-rt-btn",
            )
            rt_status = gr.Textbox(
                value="", label="", interactive=False,
                visible=True, max_lines=1, show_label=False,
                container=False,
            )

        # ── RIGHT: output ────────────────────────────────────
        with gr.Column(scale=1):
            caption_out = gr.Textbox(
                label="Description", lines=5, interactive=False,
                show_copy_button=True,
                placeholder="Description will appear here...",
                elem_id="echo-caption",
            )
            audio_out = gr.Audio(
                label="Audio", type="filepath", autoplay=True,
            )

            with gr.Row():
                repeat_btn = gr.Button("🔁 Repeat (P)", variant="secondary",
                                       elem_id="echo-repeat-btn")
                stop_btn   = gr.Button("⏹ Silence",    variant="stop",
                                       elem_id="echo-stop-btn")

            gr.HTML("""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                        border-radius:10px;padding:14px;margin-top:10px;font-size:15px;line-height:1.6">
                <strong>Tips for blind users:</strong><br>
                • <kbd>D</kbd> — describe what camera sees now<br>
                • <kbd>R</kbd> — start/stop auto-description every 3.5s<br>
                • <kbd>P</kbd> — repeat last description<br>
                • <kbd>Esc</kbd> — stop realtime<br>
                • Use <strong>Read Text</strong> to read signs, labels, screens<br>
                • Use <strong>Detect Objects</strong> to hear what's where
            </div>
            """)

    # ═══════════════════════════════════════════════════════
    #  EVENT WIRING
    # ═══════════════════════════════════════════════════════

    # Manual describe
    describe_btn.click(
        handle_describe,
        inputs=[webcam_input, task_choice, voice_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # Upload
    upload_input.change(
        handle_upload,
        inputs=[upload_input, task_choice, voice_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # ── Realtime toggle — flips gr.State, updates button label ──
    realtime_btn.click(
        toggle_realtime,
        inputs=[rt_active],
        outputs=[rt_active, realtime_btn, rt_status],
    )

    # ── Realtime streaming — fires every CAPTURE_INTERVAL seconds ──
    # Only does work when rt_active == True (gated inside handler)
    webcam_input.stream(
        handle_realtime_stream,
        inputs=[webcam_input, task_choice, voice_choice, rt_active],
        outputs=[caption_out, audio_out],
        stream_every=CAPTURE_INTERVAL,
        show_progress=False,
        time_limit=None,
    )

    # Repeat
    repeat_btn.click(
        handle_repeat,
        inputs=[voice_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # Stop
    stop_btn.click(
        handle_stop,
        inputs=[],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

if __name__ == "__main__":
    demo.launch(debug=True)
