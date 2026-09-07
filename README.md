---
title: SightLine
emoji: 
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---
# SightLine: See the World Through AI

SightLine is a voice-first multimodal AI assistant designed primarily for blind and low-vision users. Built for the **UnleashLLM Innovation Challenge** by team members Prateek Dhar Dwivedi (NIELIT Main Campus Ropar) and Lovnish Verma (NIELIT Chandigarh, Ropar Campus).

## Features
- **Real-Time Scene Understanding:** Continuously analyze the camera feed to detect people, vehicles, and obstacles using Microsoft's Florence-2 architecture.
- **Interactive VQA (Visual Question Answering):** Users can actively ask specific questions about their environment (e.g., "Where are my keys?", "Is the traffic light green?").
- **Hyper-Local & Multilingual Support (Hindi):** Automatically translates English scene descriptions into native Hindi and outputs audio using native Indic Text-to-Speech (TTS), making the tool accessible to local demographics.
- **Smart Scene Change Detection:** Triggers descriptions only when meaningful changes occur in the scene to avoid spamming the user.
- **Conversational Context:** Remembers recent objects and OCR text to provide natural context-aware responses.
- **Accessibility First:** High contrast, ARIA labels, screen reader support, large controls, and a fully hands-free experience.

## Project Structure
```text
sightline/
 src/
    main.py
    config.py
    vision/          # Florence-2 & PaddleOCR backend
    speech/          # Edge-TTS & audio queuing
    conversation/    # Context tracking & Translation fallback layer
    camera/
    accessibility/
    ui/              # Gradio web interface
 Dockerfile
 requirements.txt
 README.md
```

## Running Locally

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   # (For Windows: .\venv\Scripts\activate)
   
   pip install -r requirements.txt
   ```
   *Note: If you have a GPU, install PyTorch with CUDA support for much faster inference:*
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall
   ```

2. Start the application:
   ```bash
   python -m src.main
   ```
3. Open the UI at `http://localhost:7860`.

## Docker Deployment
```bash
docker build -t sightline .
docker run -p 7860:7860 sightline
```

## Privacy & Safety
SightLine does not permanently store camera frames or microphone recordings. It processes imagery purely to provide instant auditory feedback to the user. 

**Disclaimer:** SightLine is an AI prototype and is not a certified replacement for a cane, guide dog, caregiver, or professional mobility aid.
