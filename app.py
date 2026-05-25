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
import base64
import os
import re
from io import BytesIO
from collections import Counter

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
CAPTURE_MS = 3500
HASH_THRESHOLD = 0.12          # dHash distance to treat as "same scene"
MAX_DIM = 768                  # downscale before inference

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
#  DEDICATED TTS EVENT LOOP  (Bug-fix #4: avoids run() crashes)
# ═══════════════════════════════════════════════════════════════
_tts_loop = asyncio.new_event_loop()

def _run_tts_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=_run_tts_loop, args=(_tts_loop,), daemon=True).start()

def text_to_speech(text: str, voice_id: str = "en-US-AriaNeural") -> str | None:
    """Generate TTS audio file. Returns path or None on error."""
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
#  STATE  (Bug-fix #3: cache includes task, Bug-fix #4: cleanup)
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
        """Update cache and clean up old temp file."""
        with self.lock:
            self._cleanup()
            if self.audio and os.path.exists(self.audio):
                try: os.unlink(self.audio)
                except OSError: pass
            self.hash = h
            self.task = task
            self.text = text
            self.audio = audio
            if audio:
                self._tmp_files.append((audio, time.time()))

    def _cleanup(self):
        """Delete temp MP3s older than 5 minutes."""
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
        """Check if cached result is valid for this hash+task."""
        return (self.hash is not None
                and self.hash == h
                and self.task == task
                and self.text)

state = State()

# ═══════════════════════════════════════════════════════════════
#  IMAGE HELPERS  (Bug-fix #5: downscale, Bug-fix #6: dHash)
# ═══════════════════════════════════════════════════════════════

def dhash(img: Image.Image, size: int = 16) -> bytes:
    """Difference hash — perceptual, detects scene changes not pixel noise."""
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
    """Downscale large images to save GPU/CPU time."""
    w, h = img.size
    if max(w, h) <= MAX_DIM:
        return img
    scale = MAX_DIM / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

# ═══════════════════════════════════════════════════════════════
#  CORE INFERENCE  (Bug-fix #1: returns str, Bug-fix #6: spatial OD)
# ═══════════════════════════════════════════════════════════════

def run_inference(image: Image.Image, task_label: str) -> str:
    """Run Florence-2 and return formatted caption string."""
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
        return _format_od(result.get("<OD>", ""))
    elif task == "<OCR>":
        text_found = result.get("<OCR>", "").strip()
        return f"Text found: {text_found}" if text_found else "No text detected."
    else:
        return result.get(task, "").strip()


def _format_od(raw: str) -> str:
    """
    Florence-2 OD: person<loc_10><loc_20><loc_500><loc_400>chair<loc_...>
    → "I see a person on the left, and a chair in the center."
    """
    if not raw:
        return "No objects detected."

    tokens = re.split(r"(<loc_\d+>)", raw)
    objects: list[tuple[str, str]] = []
    label = ""
    locs: list[int] = []

    for tok in tokens:
        if tok.startswith("<loc_"):
            locs.append(int(tok[5:-1]))
            if len(locs) == 4:
                x1, _, x2, _ = locs
                cx = (x1 + x2) / 2
                if cx < 333:
                    pos = "on the left"
                elif cx < 666:
                    pos = "in the center"
                else:
                    pos = "on the right"
                objects.append((label.strip(), pos))
                locs = []
        else:
            if locs and label.strip():
                objects.append((label.strip(), ""))
            label = tok
            locs = []

    if not objects:
        return raw.strip()

    # De-duplicate
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for lbl, pos in objects:
        key = lbl.lower()
        if key and key not in seen:
            seen.add(key)
            unique.append((lbl, pos))

    if len(unique) == 1:
        lbl, pos = unique[0]
        return f"I see {lbl} {pos}." if pos else f"I see {lbl}."

    parts = [f"{lbl} {pos}".strip() for lbl, pos in unique]
    if len(parts) <= 6:
        return "I see " + ", ".join(parts[:-1]) + f", and {parts[-1]}."
    return (
        f"I see {len(unique)} objects: "
        + ", ".join(parts[:5]) + f", and {len(unique) - 5} more."
    )

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

    # Check cache
    with state.lock:
        if state.matches(h, task_key):
            yield state.text, state.audio
            return

    caption = run_inference(image, task_label)

    # Stream words for visual feedback
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


def handle_frame(b64: str, task_label: str, voice_name: str):
    """Called by JS timer — receives webcam frame as base64."""
    if not b64 or "," not in b64:
        return gr.update(), gr.update()
    try:
        data = base64.b64decode(b64.split(",", 1)[1])
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"Frame decode error: {e}")
        return gr.update(), gr.update()

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
    """Re-speak last description."""
    with state.lock:
        text = state.text
    if not text:
        return gr.update(), gr.update(value=None)
    voice_id = VOICE_MAP.get(voice_name, "en-US-AriaNeural")
    audio = text_to_speech(text, voice_id)
    return text, audio


def handle_stop():
    """Silence."""
    return gr.update(value=""), gr.update(value=None)

# ═══════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════
CSS = """
/* ── Accessibility base ── */
body { font-size: 18px !important; }
.gr-button {
    min-height: 52px !important;
    font-size: 17px !important;
    border-radius: 12px !important;
    cursor: pointer;
    transition: transform 0.08s;
}
.gr-button:active { transform: scale(0.97); }
.gr-button:focus-visible {
    outline: 3px solid #2563eb !important;
    outline-offset: 3px !important;
}
.gr-textbox textarea {
    font-size: 20px !important;
    line-height: 1.7 !important;
}
.gr-radio:focus-visible,
.gr-dropdown:focus-visible {
    outline: 3px solid #2563eb !important;
    outline-offset: 2px !important;
}

/* ── High contrast ── */
body.hc { filter: contrast(1.6) brightness(1.1); }

/* ── Font-size classes ── */
body.fs-large  .gr-textbox textarea { font-size: 26px !important; }
body.fs-large  .gr-button { font-size: 20px !important; }
body.fs-xlarge .gr-textbox textarea { font-size: 32px !important; }
body.fs-xlarge .gr-button { font-size: 22px !important; min-height: 60px !important; }

/* ── Status banner ── */
#echo-status {
    background: #1e293b; color: #f8fafc;
    padding: 12px 18px; border-radius: 10px;
    font-size: 17px; margin-bottom: 12px;
    min-height: 44px; font-weight: 600;
}

/* ── Realtime indicator ── */
#rt-indicator {
    display: inline-block; width: 14px; height: 14px;
    border-radius: 50%; background: #6b7280;
    margin-right: 8px; vertical-align: middle;
    transition: background 0.3s;
}
#rt-indicator.active {
    background: #22c55e;
    animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(34,197,94,0.5)} 50%{opacity:.5;box-shadow:0 0 0 8px rgba(34,197,94,0)} }

/* ── Mobile ── */
@media (max-width: 768px) {
    .gr-button { width: 100% !important; min-height: 58px !important; font-size: 19px !important; }
    #echo-status { font-size: 15px; }
}

/* ── Reduced motion ── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
"""

# ═══════════════════════════════════════════════════════════════
#  JAVASCRIPT  (Bug-fix #2: use input event, not MutationObserver)
# ═══════════════════════════════════════════════════════════════
JS_INIT = ""  # JS moved to gr.Blocks(js=)

# ═══════════════════════════════════════════════════════════════
#  BUILD UI
# ═══════════════════════════════════════════════════════════════
_ECHO_JS = r"""
(function(){
    let timer = null;
    let running = false;

    /* ── helpers ── */
    function setTA(el, val){
        const s = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set;
        s.call(el, val);
        el.dispatchEvent(new Event('input',{bubbles:true}));
    }
    function status(msg){
        const el = document.getElementById('echo-status');
        if(el){ el.textContent = msg; el.setAttribute('aria-label', msg); }
    }

    /* ── capture one frame → hidden textarea → hidden button ── */
    function capture(){
        const video = Array.from(document.querySelectorAll('video'))
                          .find(v => v.videoWidth > 0 && v.readyState >= 2);
        if(!video){ status('Camera not active yet.'); return; }

        const c = document.createElement('canvas');
        /* Downscale in browser to reduce base64 size ~60% */
        const s = Math.min(1, 640 / video.videoWidth);
        c.width  = Math.round(video.videoWidth  * s);
        c.height = Math.round(video.videoHeight * s);
        c.getContext('2d').drawImage(video, 0, 0, c.width, c.height);
        const b64 = c.toDataURL('image/jpeg', 0.72);

        /* FIX: use the dedicated elem_id instead of column selector */
        const ta = document.querySelector('#echo-frame-box textarea');
        if(!ta){ console.warn('[EchoLens] frame-box not found'); return; }
        setTA(ta, b64);

        setTimeout(()=>{
            /* FIX: target button by its own elem_id, not parent column */
            const btn = document.querySelector('#echo-frame-btn button');
            if(btn) btn.click();
            else console.warn('[EchoLens] frame-btn not found');
        }, 120);
    }

    /* ── toggle realtime ── */
    function echoStart(){
        if(running) return;
        running = true;
        const ind = document.getElementById('rt-indicator');
        if(ind) ind.classList.add('active');
        status('Realtime ON — describing every 3.5 seconds');
        capture();
        timer = setInterval(capture, 3500);
    }
    function echoStop(){
        if(!running) return;
        running = false;
        clearInterval(timer); timer = null;
        const ind = document.getElementById('rt-indicator');
        if(ind) ind.classList.remove('active');
        status('Realtime stopped.');
    }
    function echoToggle(){
        running ? echoStop() : echoStart();
    }

    /* FIX: expose globals immediately so onclick="echoToggle()" works */
    window.echoStart  = echoStart;
    window.echoStop   = echoStop;
    window.echoToggle = echoToggle;

    /* ── accessibility controls ── */
    window.echoFontSize = function(size){
        document.body.classList.remove('fs-large','fs-xlarge');
        if(size !== 'normal') document.body.classList.add('fs-'+size);
    };
    window.echoContrast = function(){
        document.body.classList.toggle('hc');
    };

    /* ── keyboard shortcuts ── */
    document.addEventListener('keydown', e => {
        if(e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA'
           || e.target.tagName === 'SELECT') return;
        if(e.key === 'd' || e.key === 'D')
            document.querySelector('#echo-describe-btn button')?.click();
        if(e.key === 'r' || e.key === 'R') window.echoToggle();
        if(e.key === 'p' || e.key === 'P')
            document.querySelector('#echo-repeat-btn button')?.click();
        if(e.key === 'Escape') window.echoStop();
    });

    /* ── BUG-FIX #2: announce captions via input event listener ──
       MutationObserver does NOT fire when you set .value via the
       descriptor trick. We must listen for the 'input' event instead. */
    window.addEventListener('load', () => {
        setTimeout(() => {
            const liveEl = document.getElementById('echo-live');
            const ta = document.querySelector('#echo-caption textarea');
            if (!ta || !liveEl) return;

            let lastAnnounced = '';
            ta.addEventListener('input', () => {
                const v = ta.value;
                if (!v || v === lastAnnounced) return;
                if (v.length < 8) return;
                if (v.startsWith('Please') || v.startsWith('Analyzing')) return;
                lastAnnounced = v;
                liveEl.textContent = '';
                requestAnimationFrame(() => {
                    liveEl.textContent = 'Description: ' + v.substring(0, 300);
                });
            });
        }, 2000);
    });

    /* ── auto-ready message ── */
    setTimeout(() => {
        const ta = document.querySelector('#echo-status');
        if (ta && ta.textContent.includes('ready')) {
            status('Ready — press D to describe, R to start realtime.');
        }
    }, 18000);
})();
"""

with gr.Blocks(
    title="EchoLens — Vision Assistant for the Blind",
    css=CSS,
    js=_ECHO_JS,
    theme=gr.themes.Soft(),
) as demo:

    # Screen-reader live region (Bug-fix #2: this is what actually works)
    gr.HTML('<div id="echo-live" aria-live="assertive" aria-atomic="true" '
            'style="position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden" '
            'role="status"></div>')

    gr.HTML(JS_INIT)

    # Status banner
    gr.HTML("""
    <div id="echo-status" role="status" aria-live="polite" aria-atomic="true">
        Loading model, please wait...
    </div>
    """)

    gr.Markdown("# 👁️ EchoLens — Vision Assistant")
    gr.Markdown("Helping blind and visually impaired users understand their surroundings.")

    # Accessibility toolbar
    gr.HTML("""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;" role="toolbar" aria-label="Accessibility controls">
        <button onclick="echoFontSize('normal')"
            style="padding:8px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer;font-size:14px"
            aria-label="Normal font size">A</button>
        <button onclick="echoFontSize('large')"
            style="padding:8px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer;font-size:17px"
            aria-label="Large font size">A+</button>
        <button onclick="echoFontSize('xlarge')"
            style="padding:8px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer;font-size:20px"
            aria-label="Extra large font size">A++</button>
        <button onclick="echoContrast()"
            style="padding:8px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer;font-size:14px;background:#1e293b;color:white"
            aria-label="Toggle high contrast">⬛ High Contrast</button>
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
                elem_id="echo-webcam", height=224,
            )
            upload_input = gr.Image(
                label="Upload Image", type="numpy", sources=["upload"],
                elem_id="echo-upload", height=160,
            )
            task_choice = gr.Radio(
                choices=list(TASKS.keys()),
                value="Quick Caption",
                label="What should I do?",
                elem_id="echo-task",
            )
            voice_choice = gr.Dropdown(
                choices=list(VOICE_MAP.keys()),
                value="Aria (Female, US)",
                label="Voice",
                elem_id="echo-voice",
            )

            describe_btn = gr.Button(
                "Describe Now (D)", variant="primary", size="lg",
                elem_id="echo-describe-btn",
            )

            # Realtime toggle (HTML for custom styling + indicator)
            gr.HTML("""
            <button onclick="echoToggle()"
                aria-label="Toggle realtime description every 3.5 seconds"
                style="width:100%;padding:14px;margin-top:8px;
                       background:#0f172a;color:white;border:none;
                       border-radius:12px;font-size:17px;cursor:pointer;
                       transition:background 0.2s;"
                onmouseenter="this.style.background='#1e293b'"
                onmouseleave="this.style.background='#0f172a'">
                <span id="rt-indicator"></span>
                Toggle Realtime (R)
            </button>
            """)

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
                elem_id="echo-audio",
            )

            with gr.Row():
                repeat_btn = gr.Button(
                    "Repeat (P)", variant="secondary",
                    elem_id="echo-repeat-btn",
                )
                stop_btn = gr.Button(
                    "Silence", variant="stop",
                    elem_id="echo-stop-btn",
                )

            gr.HTML("""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                        border-radius:10px;padding:14px;margin-top:10px;font-size:15px;line-height:1.6">
                <strong>Tips for blind users:</strong><br>
                • <kbd>D</kbd> — describe what camera sees now<br>
                • <kbd>R</kbd> — start/stop auto-description every 3.5s<br>
                • <kbd>P</kbd> — repeat last description<br>
                • <kbd>Esc</kbd> — stop realtime<br>
                • Use <strong>Read Text</strong> to read signs, labels, screens<br>
                • Use <strong>Detect Objects</strong> to hear what's where (left/center/right)
            </div>
            """)

    # ── Hidden plumbing (JS → Python frame relay) ────────────
    with gr.Row(visible=False):
        frame_box = gr.Textbox(elem_id="echo-frame-box", label="fb")
        frame_btn = gr.Button("go", elem_id="echo-frame-btn")

    # ═══════════════════════════════════════════════════════
    #  EVENT WIRING
    # ═══════════════════════════════════════════════════════
    describe_btn.click(
        handle_describe,
        inputs=[webcam_input, task_choice, voice_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    upload_input.change(
        handle_upload,
        inputs=[upload_input, task_choice, voice_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    frame_btn.click(
        handle_frame,
        inputs=[frame_box, task_choice, voice_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
        queue=True,
    )

    repeat_btn.click(
        handle_repeat,
        inputs=[voice_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    stop_btn.click(
        handle_stop,
        inputs=[],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

if __name__ == "__main__":
    demo.launch(debug=True)