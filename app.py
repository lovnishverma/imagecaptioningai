import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from gtts import gTTS
import tempfile

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Florence-2-base',
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device).eval()

processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)


def generate_caption_stream(image):
    if image is None:
        yield "Please upload or capture an image.", None
        return

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    inputs = processor(
        text="<MORE_DETAILED_CAPTION>",
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
        )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(image.width, image.height),
    )
    caption = result["<MORE_DETAILED_CAPTION>"]

    # Stream word by word, no audio yet
    words = caption.split()
    partial = ""
    for word in words:
        partial += ("" if partial == "" else " ") + word
        yield partial, None

    # Final yield: generate and return audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        gTTS(text=caption, lang="en", slow=False).save(tmp.name)
        audio_path = tmp.name

    print(f"\nFinal caption: {caption}")
    yield caption, audio_path


with gr.Blocks(title="EchoLens RT") as demo:
    gr.Markdown("# 👁️ EchoLens — Realtime Captioning + Speech")
    gr.Markdown("Upload or capture an image. Caption streams live then plays aloud.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Image",
                type="numpy",
                sources=["upload", "webcam"],
            )
            btn = gr.Button("Describe Image ▶", variant="primary")

        with gr.Column(scale=1):
            caption_out = gr.Textbox(
                label="Caption",
                lines=5,
                interactive=False,
                show_copy_button=True,
            )
            audio_out = gr.Audio(
                label="Audio",
                type="filepath",
                autoplay=True,
            )

    btn.click(
        fn=generate_caption_stream,
        inputs=image_input,
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

    image_input.change(
        fn=generate_caption_stream,
        inputs=image_input,
        outputs=[caption_out, audio_out],
        show_progress=False,
    )

if __name__ == "__main__":
    demo.launch(debug=True)