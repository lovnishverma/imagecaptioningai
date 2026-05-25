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
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Florence-2-base',
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device).eval()

processor = AutoProcessor.from_pretrained(
    'microsoft/Florence-2-base',
    trust_remote_code=True
)

# ── Warmup ──────────────────────────────────────────────────
_warmup_done = threading.Event()

def warmup():
    dummy = Image.new("RGB", (224, 224), color=128)
    inp = processor(text="<CAPTION>", images=dummy, return_tensors="pt").to(device)
    with torch.inference_mode():
        model.generate(input_ids=inp["input_ids"],
                       pixel_values=inp["pixel_values"],
                       max_new_tokens=20, num_beams=1)
    _warmup_done.set()
    print("Model warmed up!")

threading.Thread(target=warmup, daemon=True).start()

# ── TTS ─────────────────────────────────────────────────────
_tts_loop = asyncio.new_event_loop()

def _run_tts_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=_run_tts_loop, args=(_tts_loop,), daemon=True).start()

def text_to_speech(text: str) -> str | None:
    try:
        async def _gen():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                path = f.name
            communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural", rate="+5%")
            await communicate.save(path)
            return path
        future = asyncio.run_coroutine_threadsafe(_gen(), _tts_loop)
        return future.result(timeout=15)
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# ── Caption cache ────────────────────────────────────────────
last = {"hash": None, "text": "", "audio": None}

def image_hash(img: Image.Image) -> int:
    return hash(img.resize((16, 16)).convert("L").tobytes())

# ── Core inference ───────────────────────────────────────────
TASKS = {
    "Describe Scene":   "<MORE_DETAILED_CAPTION>",
    "Quick Caption":    "<CAPTION>",
    "Read Text (OCR)":  "<OCR>",
    "Detect Objects":   "<OD>",
}

def run_inference(image: Image.Image, task_label: str) -> tuple[str, str | None]:
    task = TASKS.get(task_label, "<CAPTION>")
    max_tok = 200 if task == "<MORE_DETAILED_CAPTION>" else 100

    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_tok,
            do_sample=False,
            num_beams=1,
        )
    raw = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(raw, task=task,
                                               image_size=(image.width, image.height))

    if task == "<OD>":
        bboxes = result.get("<OD>", {})
        labels = bboxes.get("labels", [])
        if labels:
            from collections import Counter
            counts = Counter(labels)
            caption = "I can see: " + ", ".join(
                f"{v} {k}" for k, v in counts.most_common()
            )
        else:
            caption = "No objects detected."
    elif task == "<OCR>":
        text_found = result.get("<OCR>", "").strip()
        caption = f"Text found: {text_found}" if text_found else "No text detected."
    else:
        caption = result.get(task, "").strip()

    return caption


def describe_image(image: Image.Image, task_label: str, force: bool = False):
    h = image_hash(image)
    if not force and h == last["hash"] and last["text"]:
        return last["text"], last["audio"]

    caption = run_inference(image, task_label)
    audio   = text_to_speech(caption)

    # cleanup old temp file
    if last["audio"] and os.path.exists(last["audio"]):
        try: os.unlink(last["audio"])
        except: pass

    last["hash"]  = h
    last["text"]  = caption
    last["audio"] = audio
    return caption, audio


# ── Gradio handlers ──────────────────────────────────────────
def handle_describe(image, task_label):
    """Manual describe — streams words then returns audio."""
    if image is None:
        yield "Please open the camera or upload an image.", None
        return
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    caption = run_inference(image, task_label)
    words, partial = caption.split(), ""
    for w in words:
        partial += ("" if not partial else " ") + w
        yield partial, None

    audio = text_to_speech(caption)
    last.update(hash=image_hash(image), text=caption, audio=audio)
    yield caption, audio


def handle_frame(b64: str, task_label: str):
    """Called by JS timer — receives webcam frame as base64."""
    if not b64 or "," not in b64:
        return gr.update(), gr.update()
    try:
        data = base64.b64decode(b64.split(",")[1])
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"Frame decode error: {e}")
        return gr.update(), gr.update()

    caption, audio = describe_image(image, task_label)
    if not caption:
        return gr.update(), gr.update()
    return caption, audio


def handle_upload(image, task_label):
    yield from handle_describe(image, task_label)


# ── UI ───────────────────────────────────────────────────────
CSS = """
/* ── Accessibility base ── */
body { font-size: 18px !important; }
.gr-button { min-height: 52px !important; font-size: 17px !important; }
.gr-textbox textarea { font-size: 18px !important; line-height: 1.7 !important; }

/* ── High contrast toggle support ── */
body.hc { filter: contrast(1.6) brightness(1.1); }

/* ── Font-size classes ── */
body.fs-large  * { font-size: 1.3em !important; }
body.fs-xlarge * { font-size: 1.6em !important; }

/* ── Status banner ── */
#echo-status {
    background: #1e293b; color: #f8fafc;
    padding: 10px 16px; border-radius: 8px;
    font-size: 16px; margin-bottom: 8px;
    min-height: 40px;
}

/* ── Realtime indicator ── */
#rt-indicator {
    display:inline-block; width:12px; height:12px;
    border-radius:50%; background:#6b7280;
    margin-right:8px; vertical-align:middle;
    transition: background 0.3s;
}
#rt-indicator.active { background:#22c55e; animation: pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
"""

JS_INIT = """
<script>
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
        if(!video){ status('⚠ Camera not active yet.'); return; }

        const c = document.createElement('canvas');
        c.width = video.videoWidth; c.height = video.videoHeight;
        c.getContext('2d').drawImage(video,0,0);
        const b64 = c.toDataURL('image/jpeg', 0.8);

        const ta = document.querySelector('#echo-frame-box textarea');
        if(!ta){ console.warn('[EchoLens] frame-box not found'); return; }
        setTA(ta, b64);

        setTimeout(()=>{
            const btn = document.querySelector('#echo-frame-btn button');
            if(btn) btn.click();
            else console.warn('[EchoLens] frame-btn not found');
        }, 100);
    }

    /* ── toggle realtime ── */
    window.echoStart = function(){
        if(running) return;
        running = true;
        document.getElementById('rt-indicator')?.classList.add('active');
        status('🟢 Realtime ON — describing every 3 seconds');
        capture();
        timer = setInterval(capture, 3500);
    };
    window.echoStop = function(){
        if(!running) return;
        running = false;
        clearInterval(timer); timer = null;
        document.getElementById('rt-indicator')?.classList.remove('active');
        status('⏹ Realtime stopped.');
    };
    window.echoToggle = function(){
        running ? window.echoStop() : window.echoStart();
    };

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
        if(e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if(e.key === 'r' || e.key === 'R') window.echoToggle();
        if(e.key === 'd' || e.key === 'D'){
            document.querySelector('#echo-describe-btn button')?.click();
        }
        if(e.key === 'Escape') window.echoStop();
    });

    /* ── announce captions to screen readers via aria-live ── */
    const liveRegion = document.createElement('div');
    liveRegion.setAttribute('aria-live','assertive');
    liveRegion.setAttribute('aria-atomic','true');
    liveRegion.style.cssText = 'position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden';
    liveRegion.id = 'echo-live';
    document.body.appendChild(liveRegion);

    /* Watch caption textbox and announce changes */
    const captionObserver = new MutationObserver(()=>{
        const ta = document.querySelector('#echo-caption textarea');
        if(ta && ta.value){
            document.getElementById('echo-live').textContent = ta.value;
        }
    });
    window.addEventListener('load', ()=>{
        setTimeout(()=>{
            const ta = document.querySelector('#echo-caption textarea');
            if(ta) captionObserver.observe(ta, {attributes:true,childList:true,subtree:true,characterData:true});
        }, 2000);
    });
})();
</script>
"""


with gr.Blocks(
    title="EchoLens — Vision Assistant for the Blind",
    css=CSS,
    theme=gr.themes.Soft(),
) as demo:

    gr.HTML(JS_INIT)

    # ── Accessible status banner ──
    gr.HTML("""
    <div id="echo-status" role="status" aria-live="polite" aria-atomic="true">
        EchoLens ready. Press D to describe, R to toggle realtime.
    </div>
    """)

    gr.Markdown("# 👁️ EchoLens — Vision Assistant")
    gr.Markdown("Helping blind and visually impaired users understand their surroundings.")

    # ── Accessibility toolbar ──
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
            Shortcuts: <kbd>D</kbd> describe &nbsp; <kbd>R</kbd> realtime &nbsp; <kbd>Esc</kbd> stop
        </span>
    </div>
    """)

    with gr.Row():
        # ── LEFT column ──────────────────────────────────────
        with gr.Column(scale=1):
            webcam_input = gr.Image(
                label="Camera",
                type="numpy",
                sources=["webcam"],
                elem_id="echo-webcam",
            )
            upload_input = gr.Image(
                label="Upload Image",
                type="numpy",
                sources=["upload"],
                elem_id="echo-upload",
            )
            task_choice = gr.Radio(
                choices=list(TASKS.keys()),
                value="Quick Caption",
                label="What should I do?",
                elem_id="echo-task",
            )

            # Describe once
            describe_btn = gr.Button(
                "📸 Describe (D)",
                variant="primary",
                size="lg",
                elem_id="echo-describe-btn",
            )

            # Realtime toggle
            gr.HTML("""
            <button
                onclick="echoToggle()"
                aria-label="Toggle realtime description every 3 seconds"
                style="width:100%;padding:14px;margin-top:8px;
                       background:#0f172a;color:white;border:none;
                       border-radius:8px;font-size:17px;cursor:pointer;">
                <span id="rt-indicator"></span>
                ▶ / ⏹ Toggle Realtime (R)
            </button>
            """)

        # ── RIGHT column ─────────────────────────────────────
        with gr.Column(scale=1):
            caption_out = gr.Textbox(
                label="Description",
                lines=6,
                interactive=False,
                show_copy_button=True,
                placeholder="Description will appear here...",
                elem_id="echo-caption",
            )
            audio_out = gr.Audio(
                label="Audio",
                type="filepath",
                autoplay=True,
                elem_id="echo-audio",
            )
            gr.HTML("""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                        border-radius:8px;padding:12px;margin-top:8px;font-size:14px">
                <strong>Tips for blind users:</strong><br>
                • <kbd>D</kbd> — describe what camera sees<br>
                • <kbd>R</kbd> — start/stop auto-description every 3s<br>
                • <kbd>Esc</kbd> — stop realtime<br>
                • Use <strong>Read Text</strong> mode to read signs/documents<br>
                • Use <strong>Detect Objects</strong> to count items in scene
            </div>
            """)

    # ── Hidden plumbing for JS frame passing ─────────────────
    with gr.Row(visible=False):
        frame_box = gr.Textbox(elem_id="echo-frame-box", label="fb")
        with gr.Column(elem_id="echo-frame-btn"):
            frame_btn = gr.Button("go", elem_id="echo-frame-btn-inner")

    # ── Events ───────────────────────────────────────────────
    describe_btn.click(
        fn=handle_describe,
        inputs=[webcam_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    upload_input.change(
        fn=handle_upload,
        inputs=[upload_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    frame_btn.click(
        fn=handle_frame,
        inputs=[frame_box, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
        queue=True,
    )

if __name__ == "__main__":
    demo.launch(debug=True)