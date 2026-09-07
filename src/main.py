import gradio as gr
from src.ui.components import build_ui
from src.conversation.assistant import SightLineAssistant
from src.config import CONFIG
from src.ui.styles import CSS

import threading

def main():
    print(f"Starting {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}")
    
    assistant = SightLineAssistant()
    # Load model synchronously for ZeroGPU compatibility
    assistant.initialize()
    
    demo = build_ui(assistant)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        ssr_mode=False
    )


if __name__ == "__main__":
    main()
