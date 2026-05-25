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

def warmup():
    dummy = Image.new("RGB", (224, 224), color=128)
    inp = processor(text="<CAPTION>", images=dummy, return_tensors="pt").to(device)
    with torch.inference_mode():
        model.generate(
            input_ids=inp["input_ids"],
            pixel_values=inp["pixel_values"],
            max_new_tokens=20,
            num_beams=1,
        )
    print("Model warmed up!")

threading.Thread(target=warmup, daemon=True).start()

last_caption = {"text": "", "hash": None}

def image_hash(image: Image.Image) -> int:
    thumb = image.resize((16, 16)).convert("L")
    return hash(thumb.tobytes())

async def _tts_async(text: str, path: str):
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural", rate="+10%")
    await communicate.save(path)

def text_to_speech(text: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        path = tmp.name
    asyncio.run(_tts_async(text, path))
    return path


def describe_frame(frame_b64: str, task_choice: str):
    """Called every 3s by JS via hidden button. Receives raw base64 jpeg."""
    if not frame_b64 or frame_b64 == "none" or "," not in frame_b64:
        return gr.update(), gr.update()
    try:
        img_bytes = base64.b64decode(frame_b64.split(",")[1])
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"Decode error: {e}")
        return gr.update(), gr.update()

    h = image_hash(image)
    if h == last_caption["hash"] and last_caption["text"]:
        print("Same frame, skipping.")
        return gr.update(), gr.update()

    task_map = {
        "Quick (faster)": "<CAPTION>",
        "Detailed (slower)": "<MORE_DETAILED_CAPTION>",
    }
    task = task_map.get(task_choice, "<CAPTION>")

    t0 = time.time()
    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=60 if task == "<CAPTION>" else 150,
            do_sample=False,
            num_beams=1,
        )
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        generated_text, task=task,
        image_size=(image.width, image.height),
    )
    caption = result[task]
    print(f"[{time.time()-t0:.2f}s] {caption}")

    last_caption["text"] = caption
    last_caption["hash"] = h

    audio_path = text_to_speech(caption)
    return caption, audio_path


def describe_once(image, task_choice):
    """Manual describe with word streaming."""
    if image is None:
        yield "Please open the camera first.", None
        return
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    task_map = {
        "Quick (faster)": "<CAPTION>",
        "Detailed (slower)": "<MORE_DETAILED_CAPTION>",
    }
    task = task_map.get(task_choice, "<CAPTION>")

    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=60 if task == "<CAPTION>" else 150,
            do_sample=False,
            num_beams=1,
        )
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        generated_text, task=task,
        image_size=(image.width, image.height),
    )
    caption = result[task]
    last_caption["text"] = caption
    last_caption["hash"] = image_hash(image)

    for word in caption.split():
        yield (caption[:caption.index(word) + len(word)]), None

    audio_path = text_to_speech(caption)
    yield caption, audio_path


def describe_upload(image, task_choice):
    """Upload image describe with streaming."""
    if image is None:
        yield "Please upload an image.", None
        return
    yield from describe_once(image, task_choice)


WEBCAM_JS = """
<script>
(function() {
    let realtimeTimer = null;
    let isRunning = false;

    function getVideo() {
        // Gradio renders webcam inside shadow DOM or iframe — try all videos
        const videos = document.querySelectorAll('video');
        for (const v of videos) {
            if (v.videoWidth > 0 && v.readyState >= 2) return v;
        }
        return null;
    }

    function setNativeValue(el, value) {
        const setter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype, 'value'
        ).set;
        setter.call(el, value);
        el.dispatchEvent(new Event('input', { bubbles: true }));
    }

    function captureAndSend() {
        const video = getVideo();
        if (!video) {
            console.warn('[EchoLens] No active video found');
            return;
        }
        const canvas = document.createElement('canvas');
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const b64 = canvas.toDataURL('image/jpeg', 0.75);

        const box = document.querySelector('#frame-box textarea');
        if (!box) { console.warn('[EchoLens] No frame-box textarea'); return; }
        setNativeValue(box, b64);

        setTimeout(() => {
            const btn = document.querySelector('#frame-btn button');
            if (btn) {
                btn.click();
                console.log('[EchoLens] Frame sent');
            } else {
                console.warn('[EchoLens] No frame-btn button');
            }
        }, 150);
    }

    window.echoToggleRealtime = function() {
        const toggleBtn = document.querySelector('#rt-btn button');
        if (!isRunning) {
            isRunning = true;
            if (toggleBtn) {
                toggleBtn.textContent = '⏹ Stop Realtime';
                toggleBtn.style.background = '#ef4444';
                toggleBtn.style.color = 'white';
            }
            captureAndSend();
            realtimeTimer = setInterval(captureAndSend, 3000);
            console.log('[EchoLens] Realtime ON');
        } else {
            isRunning = false;
            clearInterval(realtimeTimer);
            realtimeTimer = null;
            if (toggleBtn) {
                toggleBtn.textContent = '▶ Start Realtime';
                toggleBtn.style.background = '';
                toggleBtn.style.color = '';
            }
            console.log('[EchoLens] Realtime OFF');
        }
    };

    window.echoDescribeOnce = function() {
        captureAndSend();
    };
})();
</script>
"""


with gr.Blocks(title="EchoLens RT", theme=gr.themes.Soft()) as demo:

    gr.HTML(WEBCAM_JS)

    gr.Markdown("""
# 👁️ EchoLens — Realtime Vision Assistant
**For blind and visually impaired users.**
- Open your camera below
- Press **Describe Once** for a single description
- Press **Start Realtime** to auto-describe every 3 seconds
""")

    with gr.Row():
        # ── LEFT: camera ──────────────────────────────────────
        with gr.Column(scale=1):
            webcam_input = gr.Image(
                label="Live Camera",
                type="numpy",
                sources=["webcam"],
            )
            upload_input = gr.Image(
                label="Or Upload an Image",
                type="numpy",
                sources=["upload"],
            )
            task_choice = gr.Radio(
                choices=["Quick (faster)", "Detailed (slower)"],
                value="Quick (faster)",
                label="Caption detail",
            )
            with gr.Row():
                # These buttons call JS directly via elem_id
                gr.HTML("""
                <div style="display:flex; gap:8px; margin-top:4px;">
                    <button
                        onclick="window.echoDescribeOnce()"
                        style="flex:1; padding:10px; background:#6366f1; color:white;
                               border:none; border-radius:8px; font-size:15px; cursor:pointer;">
                        📸 Describe Once
                    </button>
                    <button
                        id="rt-btn-inner"
                        onclick="window.echoToggleRealtime()"
                        style="flex:1; padding:10px; background:#10b981; color:white;
                               border:none; border-radius:8px; font-size:15px; cursor:pointer;">
                        ▶ Start Realtime
                    </button>
                </div>
                """)

        # ── RIGHT: output ──────────────────────────────────────
        with gr.Column(scale=1):
            caption_out = gr.Textbox(
                label="Caption",
                lines=5,
                interactive=False,
                show_copy_button=True,
                placeholder="Caption will appear here...",
            )
            audio_out = gr.Audio(
                label="Audio Description",
                type="filepath",
                autoplay=True,
            )
            gr.Markdown("*Realtime mode auto-describes every 3 seconds.*")

    # ── Hidden plumbing: JS → Python ──────────────────────────
    with gr.Row(visible=False):
        frame_box = gr.Textbox(elem_id="frame-box", label="frame_box")
        with gr.Column(elem_id="frame-btn"):
            frame_btn = gr.Button("send", elem_id="frame-submit")

    # Upload describe
    upload_input.change(
        fn=describe_upload,
        inputs=[upload_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # Hidden frame button → describe_frame (realtime + describe-once via JS)
    frame_btn.click(
        fn=describe_frame,
        inputs=[frame_box, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
        queue=True,
    )

if __name__ == "__main__":
    demo.launch(debug=True)