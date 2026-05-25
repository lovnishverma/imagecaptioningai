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
    """Non-streaming version used by timer tick."""
    if image is None:
        return gr.update(), gr.update()

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    h = image_hash(image)
    if h == last_caption["hash"] and last_caption["text"]:
        return gr.update(), gr.update()  # same frame, skip

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
    """Streaming version for manual button."""
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


def realtime_tick(image, task_choice, is_active):
    if not is_active:
        return gr.update(), gr.update()
    return run_caption(image, task_choice)


with gr.Blocks(title="EchoLens RT", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ EchoLens — Realtime Vision Assistant")
    gr.Markdown("For blind and visually impaired users. Use **Start Realtime** for continuous camera description.")

    is_realtime = gr.State(False)

    with gr.Row():
        with gr.Column(scale=1):
            # Webcam-only input for streaming mode
            webcam_input = gr.Image(
                label="Live Camera",
                type="numpy",
                sources=["webcam"],
                streaming=True,                        # fix: only works with webcam alone
                webcam_options=gr.WebcamOptions(mirror_webcam=False),  # fix: replaces mirror_webcam
            )
            # Separate upload input
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

    # Manual describe — works with both webcam and upload
    btn.click(
        fn=generate_caption_stream,
        inputs=[webcam_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )
    upload_input.change(
        fn=generate_caption_stream,
        inputs=[upload_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    # Toggle realtime
    realtime_btn.click(
        fn=lambda s: (
            not s,
            gr.update(value="⏹ Stop Realtime" if not s else "▶ Start Realtime",
                      variant="stop" if not s else "secondary"),
            gr.Timer(active=not s),
        ),
        inputs=[is_realtime],
        outputs=[is_realtime, realtime_btn, timer],
    )

    # Timer tick
    timer.tick(
        fn=realtime_tick,
        inputs=[webcam_input, task_choice, is_realtime],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

if __name__ == "__main__":
    demo.launch(debug=True)