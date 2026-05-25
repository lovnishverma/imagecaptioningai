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
    if not frame_b64 or "," not in frame_b64:
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


def describe_upload(image, task_choice):
    if image is None:
        yield "Please upload an image.", None
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

    words = caption.split()
    partial = ""
    for word in words:
        partial += ("" if partial == "" else " ") + word
        yield partial, None

    audio_path = text_to_speech(caption)
    yield caption, audio_path


with gr.Blocks(title="EchoLens RT", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 👁️ EchoLens — Realtime Vision Assistant")
    gr.Markdown("**For blind and visually impaired users.** Open camera → click **Start Realtime** for auto-description every 3 seconds.")

    with gr.Row():
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
            describe_btn = gr.Button("📸 Describe Once", variant="primary")
            realtime_btn = gr.Button("▶ Start Realtime", variant="secondary", elem_id="realtime-btn")

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
            status_out = gr.Textbox(
                label="Status",
                value="Ready.",
                interactive=False,
                lines=1,
            )

    # Hidden plumbing for JS → Python frame passing
    with gr.Row(visible=False):
        frame_box  = gr.Textbox(elem_id="frame-box",  label="fb")
        frame_btn  = gr.Button("go", elem_id="frame-btn")

    # ── Gradio events ──────────────────────────────────────────

    describe_btn.click(
        fn=describe_upload,
        inputs=[webcam_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    upload_input.change(
        fn=describe_upload,
        inputs=[upload_input, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    frame_btn.click(
        fn=describe_frame,
        inputs=[frame_box, task_choice],
        outputs=[caption_out, audio_out],
        show_progress=False,
        queue=True,
    )

    # Realtime toggle — Python just flips label/color,
    # JS (below) does the actual capture loop
    realtime_btn.click(
        fn=None,
        js="""
() => {
    const btn = document.querySelector('#realtime-btn button');
    if (!btn) return;

    if (btn.dataset.running === 'true') {
        // --- STOP ---
        btn.dataset.running = 'false';
        clearInterval(window._echoTimer);
        window._echoTimer = null;
        btn.textContent = '▶ Start Realtime';
        btn.style.background = '';
        btn.style.color = '';
        const s = document.querySelector('#status-box textarea');
        if (s) { Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(s,'Realtime stopped.'); s.dispatchEvent(new Event('input',{bubbles:true})); }
    } else {
        // --- START ---
        btn.dataset.running = 'true';
        btn.textContent = '⏹ Stop Realtime';
        btn.style.background = '#ef4444';
        btn.style.color = 'white';
        const s = document.querySelector('#status-box textarea');
        if (s) { Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(s,'Realtime running...'); s.dispatchEvent(new Event('input',{bubbles:true})); }

        function capture() {
            const video = [...document.querySelectorAll('video')].find(v => v.videoWidth > 0 && v.readyState >= 2);
            if (!video) { console.warn('[EchoLens] no video'); return; }
            const c = document.createElement('canvas');
            c.width = video.videoWidth; c.height = video.videoHeight;
            c.getContext('2d').drawImage(video, 0, 0);
            const b64 = c.toDataURL('image/jpeg', 0.75);

            const box = document.querySelector('#frame-box textarea');
            if (!box) { console.warn('[EchoLens] no frame-box'); return; }
            Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(box, b64);
            box.dispatchEvent(new Event('input',{bubbles:true}));

            setTimeout(() => {
                const fb = document.querySelector('#frame-btn button');
                if (fb) fb.click();
                else console.warn('[EchoLens] no frame-btn');
            }, 200);
        }

        capture(); // immediate
        window._echoTimer = setInterval(capture, 3500);
    }
}
""",
    )

    # Status box update from JS
    status_out.change(fn=None, inputs=[], outputs=[])

    # Inject status box elem_id via HTML trick
    gr.HTML("""
    <script>
    // Patch status textarea elem_id so JS can find it
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => {
            const labels = document.querySelectorAll('.label-wrap span');
            labels.forEach(l => {
                if (l.textContent === 'Status') {
                    const ta = l.closest('.form')?.querySelector('textarea');
                    if (ta) ta.closest('.block')?.setAttribute('id','status-box');
                }
            });
        }, 2000);
    });
    </script>
    """)

if __name__ == "__main__":
    demo.launch(debug=True)