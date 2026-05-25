import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Florence-2-base',
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device).eval()

processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)

# ── JS injected into Gradio to trigger browser speech on caption update ──
SPEECH_JS = """
<script>
function speakCaption(text) {
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 1.1;
    utt.pitch = 1.0;
    window.speechSynthesis.speak(utt);
}

// Watch the caption textbox for changes, speak when it stops updating
let debounce;
const observer = new MutationObserver(() => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
        const el = document.querySelector('#caption-output textarea');
        if (el && el.value) speakCaption(el.value);
    }, 400);   // 400 ms after last token → speak
});

// Wait for DOM to be ready then attach observer
window.addEventListener('load', () => {
    const attach = () => {
        const el = document.querySelector('#caption-output textarea');
        if (el) {
            observer.observe(el, { attributes: true, childList: true, subtree: true, characterData: true });
        } else {
            setTimeout(attach, 300);
        }
    };
    attach();
});
</script>
"""


def generate_caption_stream(image):
    """Stream caption tokens live into the textbox."""
    if image is None:
        yield "Please upload or capture an image."
        return

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    inputs = processor(
        text="<MORE_DETAILED_CAPTION>",
        images=image,
        return_tensors="pt"
    ).to(device)

    # ── Greedy decode: ~3× faster than beam=3, good enough for captions ──
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,       # was 1024 — captions rarely need more
            do_sample=False,
            num_beams=1,              # greedy; was 3
        )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(image.width, image.height),
    )
    caption = result["<MORE_DETAILED_CAPTION>"]

    # ── Simulate streaming: yield word-by-word for live feel ──
    words = caption.split()
    partial = ""
    for word in words:
        partial += ("" if partial == "" else " ") + word
        yield partial

    print(f"\nFinal caption: {caption}")


with gr.Blocks(title="EchoLens RT") as demo:
    # Inject speech JS once
    gr.HTML(SPEECH_JS)

    gr.Markdown("# 👁️ EchoLens — Realtime Captioning + Speech")
    gr.Markdown("Upload or capture an image. Caption streams live and is read aloud automatically.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Image",
                type="numpy",
                sources=["upload", "webcam"],   # webcam support
            )
            btn = gr.Button("Describe Image ▶", variant="primary")

        with gr.Column(scale=1):
            caption_out = gr.Textbox(
                label="Caption",
                lines=5,
                interactive=False,
                elem_id="caption-output",      # JS watches this ID
                show_copy_button=True,
            )

    btn.click(
        fn=generate_caption_stream,
        inputs=image_input,
        outputs=caption_out,
        show_progress=False,
    )

    # Also trigger on image change for true realtime feel
    image_input.change(
        fn=generate_caption_stream,
        inputs=image_input,
        outputs=caption_out,
        show_progress=False,
    )

if __name__ == "__main__":
    demo.launch(debug=True)