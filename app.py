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
    """Core caption + TTS — used by both button and timer."""
    if image is None:
        return "", None

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    h = image_hash(image)
    if h == last_caption["hash"] and last_caption["text"]:
        return last_caption["text"], None   # same frame, skip

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
    """Streaming version for manual button — yields word by word."""
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


# ── State to track realtime mode ──
realtime_on = {"value": False}

def toggle_realtime(current_state):
    realtime_on["value"] = not realtime_on["value"]
    if realtime_on["value"]:
        return gr.update(value="⏹ Stop Realtime", variant="stop")
    else:
        return gr.update(value="▶ Start Realtime", variant="secondary")

def realtime_tick(image, task_choice, is_active):
    """Called by gr.Timer every N seconds. Only runs if realtime is active."""
    if not is_active:
        return gr.update(), gr.update()
    caption, audio = run_caption(image, task_choice)
    if not caption:
        return gr.update(), gr.update()
    return caption, audio


with gr.Blocks(title="EchoLens RT", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ EchoLens — Realtime Vision Assistant")
    gr.Markdown("For blind and visually impaired users. Use **Start Realtime** for continuous camera description.")

    # State
    is_realtime = gr.State(False)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Camera",
                type="numpy",
                sources=["webcam", "upload"],
                mirror_webcam=False,
                streaming=True,       # streams webcam frames continuously
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

    # Timer fires every 3 seconds
    timer = gr.Timer(value=3, active=False)

    # Manual describe
    btn.click(
        fn=generate_caption_stream,
        inputs=[image_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # Toggle realtime on/off
    realtime_btn.click(
        fn=lambda s: (not s, gr.update(value="⏹ Stop Realtime" if not s else "▶ Start Realtime",
                                        variant="stop" if not s else "secondary"),
                      gr.Timer(active=not s)),
        inputs=[is_realtime],
        outputs=[is_realtime, realtime_btn, timer],
    )

    # Timer tick → describe current frame
    timer.tick(
        fn=realtime_tick,
        inputs=[image_input, task_choice, is_realtime],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

if __name__ == "__main__":
    demo.launch(debug=True)