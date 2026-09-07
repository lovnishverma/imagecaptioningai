import gradio as gr

def wire_events(demo, components, assistant):
    c = components

    def handle_describe(webcam_img, upload_img, task, voice, question):
        is_webcam = webcam_img is not None
        img = webcam_img if is_webcam else upload_img
        if img is None:
            return gr.update(), gr.update(), f'<div id="sightline-status">Please provide an image.</div>'
            
        import numpy as np
        import cv2
        # If the image came from the webcam, it is likely horizontally mirrored by Gradio.
        # We must un-mirror it so text and spatial relationships are correct!
        if is_webcam and isinstance(img, np.ndarray):
            img = cv2.flip(img, 1)
            
        text, audio, status = assistant.process_image(img, task, voice, force=True, question=question)
        return text or gr.update(), audio or gr.update(), f'<div id="sightline-status">{status}</div>'

    c["describe_btn"].click(
        handle_describe,
        inputs=[c["webcam"], c["upload"], c["task_radio"], c["voice_dropdown"], c["question_box"]],
        outputs=[c["caption_box"], c["audio_player"], c["status_bar"]]
    )

    def toggle_rt(is_active):
        new_state = not is_active
        btn_text = "⏹ Stop Realtime (R)" if new_state else "▶ Start Realtime (R)"
        status_msg = "Realtime Started" if new_state else "Realtime Paused"
        return new_state, gr.update(value=btn_text), f'<div id="sightline-status">{status_msg}</div>'

    c["realtime_btn"].click(
        toggle_rt,
        inputs=[c["rt_state"]],
        outputs=[c["rt_state"], c["realtime_btn"], c["status_bar"]]
    )

    def handle_rt_stream(image, task, voice, question, is_active):
        if is_active is False:
            return gr.update(), gr.update(), f'<div id="sightline-status">Realtime Paused</div>'
        if not is_active:
            # Fallback if None
            is_active = True
            
        if image is None:
            return gr.update(), gr.update(), f'<div id="sightline-status">No Camera</div>'
            
        import numpy as np
        import cv2
        # Un-mirror the webcam feed for backend processing
        if isinstance(image, np.ndarray):
            image = cv2.flip(image, 1)
            
        text, audio, status = assistant.process_image(image, task, voice, force=False, question=question)
        return text or gr.update(), audio or gr.update(), f'<div id="sightline-status">{status}</div>'

    c["webcam"].stream(
        handle_rt_stream,
        inputs=[c["webcam"], c["task_radio"], c["voice_dropdown"], c["question_box"], c["rt_state"]],
        outputs=[c["caption_box"], c["audio_player"], c["status_bar"]],
        stream_every=3.0
    )

