import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import edge_tts
import tempfile
import asyncio
import threading
import time
import numpy as np
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


def caption_from_pil(image: Image.Image, task_choice: str):
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


def describe_frame(frame_b64: str, task_choice: str):
    """Receives base64 jpeg from JS webcam snapshot."""
    if not frame_b64 or frame_b64 == "none":
        return gr.update(), gr.update()
    try:
        # Strip data URL header if present
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",")[1]
        img_bytes = base64.b64decode(frame_b64)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        return caption_from_pil(image, task_choice)
    except Exception as e:
        print(f"Frame error: {e}")
        return gr.update(), gr.update()


def describe_upload(image, task_choice):
    """Streaming version for manual/upload."""
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


# JS captures webcam frame → puts base64 in hidden textbox → triggers hidden button
WEBCAM_JS = """
<script>
let realtimeTimer = null;

function captureAndSend() {
    const video = document.querySelector('video');
    if (!video || video.readyState < 2) {
        console.log('Video not ready');
        return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const b64 = canvas.toDataURL('image/jpeg', 0.7);

    // Put base64 into hidden textbox
    const hiddenBox = document.querySelector('#frame-input textarea');
    if (!hiddenBox) { console.log('No hidden box'); return; }

    // Set value via React/Svelte-compatible input event
    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    nativeInputValueSetter.call(hiddenBox, b64);
    hiddenBox.dispatchEvent(new Event('input', { bubbles: true }));

    // Click hidden submit button
    setTimeout(() => {
        const btn = document.querySelector('#frame-submit');
        if (btn) btn.click();
        else console.log('No submit btn');
    }, 100);
}

function startRealtime() {
    if (realtimeTimer) return;
    console.log('Realtime started');
    captureAndSend(); // immediate first capture
    realtimeTimer = setInterval(captureAndSend, 3000);
}

function stopRealtime() {
    if (realtimeTimer) {
        clearInterval(realtimeTimer);
        realtimeTimer = null;
        console.log('Realtime stopped');
    }
}

// Expose globally so Gradio buttons can call them
window.startRealtime = startRealtime;
window.stopRealtime = stopRealtime;
</script>
"""


with gr.Blocks(title="EchoLens RT", theme=gr.themes.Soft()) as demo:
    gr.HTML(WEBCAM_JS)
    gr.Markdown("# 👁️ EchoLens — Realtime Vision Assistant")
    gr.Markdown("For blind and visually impaired users. Press **Start Realtime** to auto-describe every 3 seconds.")

    is_realtime = gr.State(False)

    with gr.Row():
        with gr.Column(scale=1):
            webcam_input = gr.Image(
                label="Live Camera",
                type="numpy",
                sources=["webcam"],
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
                realtime_btn = gr.Button(
                    "▶ Start Realtime",
                    variant="secondary",
                    elem_id="realtime-toggle-btn",
                )

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
            gr.Markdown("*Realtime mode captures from webcam every 3 seconds.*")

    # Hidden components: JS writes frame here, triggers caption
    with gr.Row(visible=False):
        frame_input = gr.Textbox(
            elem_id="frame-input",
            label="frame",
        )
        frame_btn = gr.Button(
            "submit frame",
            elem_id="frame-submit",
        )

    # Manual describe once (webcam snapshot)
    btn.click(
        fn=describe_upload,
        inputs=[webcam_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # Upload image
    upload_input.change(
        fn=describe_upload,
        inputs=[upload_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # Hidden frame button → caption
    frame_btn.click(
        fn=describe_frame,
        inputs=[frame_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # Realtime toggle: update button label + call JS start/stop
    realtime_btn.click(
        fn=lambda s: (
            not s,
            gr.update(
                value="⏹ Stop Realtime" if not s else "▶ Start Realtime",
                variant="stop" if not s else "secondary",
            ),
        ),
        inputs=[is_realtime],
        outputs=[is_realtime, realtime_btn],
    ).then(
        fn=None,
        js="""
        (is_realtime) => {
            if (is_realtime) {
                window.startRealtime();
            } else {
                window.stopRealtime();
            }
        }
        """,
        inputs=[is_realtime],
    )

if __name__ == "__main__":
    demo.launch(debug=True)