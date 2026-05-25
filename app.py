import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import edge_tts
import tempfile
import asyncio
import threading
import time

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


def run_caption(image, task_choice):
    if image is None:
        return gr.update(), gr.update()

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    h = image_hash(image)
    if h == last_caption["hash"] and last_caption["text"]:
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
        generated_text,
        task=task,
        image_size=(image.width, image.height),
    )
    caption = result[task]
    print(f"Caption ({time.time()-t0:.2f}s): {caption}")

    last_caption["text"] = caption
    last_caption["hash"] = h

    audio_path = text_to_speech(caption)
    return caption, audio_path


def generate_caption_stream(image, task_choice):
    if image is None:
        yield "Please upload or capture an image.", None
        return

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    h = image_hash(image)
    if h == last_caption["hash"] and last_caption["text"]:
        audio_path = text_to_speech(last_caption["text"])
        yield last_caption["text"], audio_path
        return

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
        generated_text,
        task=task,
        image_size=(image.width, image.height),
    )
    caption = result[task]

    last_caption["text"] = caption
    last_caption["hash"] = h

    words = caption.split()
    partial = ""
    for word in words:
        partial += ("" if partial == "" else " ") + word
        yield partial, None

    audio_path = text_to_speech(caption)
    yield caption, audio_path


def realtime_tick(snapshot, task_choice, is_active):
    """Timer calls this — snapshot is set by JS auto-capture."""
    if not is_active or snapshot is None:
        return gr.update(), gr.update()
    return run_caption(snapshot, task_choice)


# ── JS: auto-snapshot webcam into hidden gr.Image every 3s ──
WEBCAM_JS = """
<script>
let realtimeInterval = null;

function startRealtimeCapture() {
    if (realtimeInterval) return;
    realtimeInterval = setInterval(() => {
        // Find the webcam video element
        const video = document.querySelector('video');
        if (!video || video.readyState < 2) return;

        // Draw frame to canvas
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Convert to blob and set on the hidden snapshot component
        canvas.toBlob((blob) => {
            const file = new File([blob], 'snapshot.jpg', { type: 'image/jpeg' });
            const dt = new DataTransfer();
            dt.items.add(file);

            // Find the snapshot upload input (second image component)
            const inputs = document.querySelectorAll('input[type=file]');
            if (inputs.length >= 2) {
                inputs[1].files = dt.files;
                inputs[1].dispatchEvent(new Event('change', { bubbles: true }));
            }
        }, 'image/jpeg', 0.8);
    }, 3000);
}

function stopRealtimeCapture() {
    if (realtimeInterval) {
        clearInterval(realtimeInterval);
        realtimeInterval = null;
    }
}

// Listen for realtime toggle button clicks
window.addEventListener('load', () => {
    const observer = new MutationObserver(() => {
        const btn = document.querySelector('button[aria-label="realtime-btn"]') ||
                    [...document.querySelectorAll('button')].find(b => b.textContent.includes('Start Realtime') || b.textContent.includes('Stop Realtime'));
        if (btn) {
            btn.addEventListener('click', () => {
                if (btn.textContent.includes('Stop')) {
                    startRealtimeCapture();
                } else {
                    stopRealtimeCapture();
                }
            });
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
"""


with gr.Blocks(title="EchoLens RT", theme=gr.themes.Soft()) as demo:
    gr.HTML(WEBCAM_JS)
    gr.Markdown("# 👁️ EchoLens — Realtime Vision Assistant")
    gr.Markdown("For blind and visually impaired users. Use **Start Realtime** for continuous camera description.")

    is_realtime = gr.State(False)

    with gr.Row():
        with gr.Column(scale=1):
            # Live webcam — user sees this
            webcam_input = gr.Image(
                label="Live Camera",
                type="numpy",
                sources=["webcam"],
                # No streaming=True — just snapshots
            )
            # Hidden: receives auto-snapshots from JS for realtime mode
            snapshot_input = gr.Image(
                label="Snapshot (auto)",
                type="numpy",
                sources=["upload"],
                visible=False,
            )
            upload_input = gr.Image(
                label="Or Upload Image",
                type="numpy",
                sources=["upload"],
            )
            task_choice = gr.Radio(
                choices=["Quick (faster)", "Detailed (slower)"],
                value="Quick (faster)",
                label="Caption mode",
            )
            with gr.Row():
                btn = gr.Button("Describe Once ▶", variant="primary")
                realtime_btn = gr.Button("▶ Start Realtime", variant="secondary")

        with gr.Column(scale=1):
            caption_out = gr.Textbox(
                label="Caption",
                lines=4,
                interactive=False,
                show_copy_button=True,
            )
            audio_out = gr.Audio(
                label="Audio Description",
                type="filepath",
                autoplay=True,
            )
            gr.Markdown("*Realtime mode describes every 3 seconds automatically.*")

    timer = gr.Timer(value=3, active=False)

    # Manual describe
    btn.click(
        fn=generate_caption_stream,
        inputs=[webcam_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # Upload triggers caption
    upload_input.change(
        fn=generate_caption_stream,
        inputs=[upload_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # Snapshot change (from JS auto-capture) triggers caption
    snapshot_input.change(
        fn=run_caption,
        inputs=[snapshot_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # Toggle realtime timer
    realtime_btn.click(
        fn=lambda s: (
            not s,
            gr.update(
                value="⏹ Stop Realtime" if not s else "▶ Start Realtime",
                variant="stop" if not s else "secondary"
            ),
            gr.Timer(active=not s),
        ),
        inputs=[is_realtime],
        outputs=[is_realtime, realtime_btn, timer],
    )

    # Timer tick — reads last webcam snapshot
    timer.tick(
        fn=realtime_tick,
        inputs=[webcam_input, task_choice, is_realtime],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

if __name__ == "__main__":
    demo.launch(debug=True)