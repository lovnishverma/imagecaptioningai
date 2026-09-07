import gradio as gr
from src.ui.styles import CSS
from src.config import CONFIG

def _status_html(msg: str) -> str:
    """Wrap a status message in the styled status-bar div."""
    return f'<div id="sightline-status" role="status" aria-live="polite">{msg}</div>'

def build_ui(assistant) -> gr.Blocks:
    auto_start_js = """
    function() {
        setInterval(function() {
            let clickTarget = function(btn) {
                let text = (btn.textContent || btn.innerText || '').toLowerCase().trim();
                if (text === 'click to access webcam' || text === 'record') {
                    btn.click();
                }
            };
            
            // Check Shadow DOMs
            document.querySelectorAll('*').forEach(function(el) {
                if (el.shadowRoot) {
                    el.shadowRoot.querySelectorAll('button').forEach(clickTarget);
                }
            });
            
            // Check regular DOM
            document.querySelectorAll('button').forEach(clickTarget);
        }, 1000); // Check every second indefinitely
    }
    """
    with gr.Blocks(title=f"{CONFIG.APP_NAME} v{CONFIG.APP_VERSION}", js=auto_start_js, css=CSS) as demo:
        gr.HTML('<div class="sr-only" aria-live="assertive" id="aria-live-region" role="status"></div>')
        
        status_bar = gr.HTML(_status_html("✅ Ready — Press D to describe"))

        gr.Markdown(f"# 👁️ {CONFIG.APP_NAME}")
        
        rt_state = gr.State(True)

        with gr.Row():
            with gr.Column(scale=1):
                # Controls at the top
                with gr.Row():
                    describe_btn = gr.Button("🔍 Describe (D)", variant="primary", elem_id="btn-describe")
                    realtime_btn = gr.Button("⚫ Stop Realtime (R)", variant="secondary", elem_id="btn-realtime")
                
                with gr.Row():
                    task_radio = gr.Radio(
                        choices=["Quick Glance", "Detailed Scene", "Immersive Description", "Read Text", "Ask Question"],
                        value="Detailed Scene",
                        label="Mode"
                    )
                    voice_dropdown = gr.Dropdown(
                        choices=["English (US) - Aria", "Hindi - Swara"],
                        value="English (US) - Aria",
                        label="Voice"
                    )
                
                question_box = gr.Textbox(
                    label="Ask a Question (for 'Ask Question' mode)",
                    placeholder="e.g. Where are my keys?",
                    lines=1
                )
                
                webcam = gr.Image(label="📷 Camera", type="numpy", sources=["webcam"], streaming=True)
                with gr.Accordion("Upload Image", open=False):
                    upload = gr.Image(label="📁 Upload", type="numpy", sources=["upload"])

            with gr.Column(scale=1):
                caption_box = gr.Textbox(label="📝 Description", lines=6, interactive=False)
                audio_player = gr.Audio(label="🔊 Audio", type="filepath", autoplay=True)
                
                with gr.Row():
                    repeat_btn = gr.Button("🔁 Repeat (P)", elem_id="btn-repeat")
                    stop_btn = gr.Button("⏹ Stop (Esc)", elem_id="btn-stop")

        components = {
            "webcam": webcam, "upload": upload, "task_radio": task_radio,
            "voice_dropdown": voice_dropdown, "question_box": question_box,
            "describe_btn": describe_btn,
            "realtime_btn": realtime_btn, "caption_box": caption_box,
            "audio_player": audio_player, "status_bar": status_bar,
            "rt_state": rt_state, "repeat_btn": repeat_btn, "stop_btn": stop_btn
        }
        
        from src.ui.events import wire_events
        wire_events(demo, components, assistant)
        
        return demo
