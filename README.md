---
title: Imagecaptioningai
emoji: 🌍
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: true
license: apache-2.0
short_description: Image to Caption
---

# 👁️ EchoLens — Realtime Vision Assistant for Blind & Low-Vision Users

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Gradio](https://img.shields.io/badge/Gradio-5.34.2-orange)](https://gradio.app)
[![Model: Florence-2](https://img.shields.io/badge/Model-Florence--2--base-purple)](https://huggingface.co/microsoft/Florence-2-base)

**EchoLens** is an accessible, real-time vision assistant that helps blind and low-vision users understand their surroundings through AI-powered image captioning and text-to-speech. Point a webcam at the world, and EchoLens describes what it sees — out loud.

---

## ✨ Features

- **🎯 Instant scene description** — Press `D` or click *Describe Now* to hear what the camera sees
- **🔄 Realtime mode** — Auto-describes the scene every 3.5 seconds; skips unchanged frames using perceptual hashing (dHash)
- **🧠 5 vision tasks** — Quick Caption, Describe Scene, Detailed Description, OCR (text reading), and Object Detection with spatial positions (left / center / right)
- **🗣️ Natural-sounding TTS** — Powered by Microsoft Edge TTS with 5 voice options across US and UK accents
- **📁 Image upload support** — Upload a photo from disk for immediate description
- **🔁 Repeat last description** — Press `P` to replay the last audio at any time
- **♿ Accessible UI** — Font size controls (A / A+ / A++), high-contrast toggle, full keyboard shortcuts, and ARIA live regions for screen-reader compatibility
- **📊 Session statistics** — Track manual describes, realtime captures, and history
- **⚡ Smart scene-change detection** — Only re-describes when the scene actually changes, saving compute and reducing audio fatigue
- **🖥️ GPU & CPU support** — Auto-detects CUDA, MPS (Apple Silicon), or CPU; optimized dtype per device

---

## 🎹 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `D` | Describe what the camera sees right now |
| `R` | Toggle realtime auto-description on/off |
| `P` | Repeat the last description |
| `Esc` | Stop all audio and exit realtime mode |

> Shortcuts work globally — no need to focus any UI element first. They are disabled while typing in text fields.

---

## 🧠 Vision Tasks

| Task | Token | Description |
|------|-------|-------------|
| **Quick Caption** | `<CAPTION>` | Short one-line summary of the scene |
| **Describe Scene** | `<DETAILED_CAPTION>` | Detailed multi-sentence description |
| **Detailed Description** | `<MORE_DETAILED_CAPTION>` | Thorough paragraph-length description |
| **Read Text (OCR)** | `<OCR>` | Reads any text visible in the image (signs, labels, screens) |
| **Detect Objects** | `<OD>` | Lists detected objects and their spatial positions |

### Object Detection Output Example

> *"I see person in the center, cup on the right, laptop on the left, and book on the left."*

The spatial positions (left / center / right) are computed from Florence-2's bounding-box coordinates, giving users a sense of *where* things are, not just *what* things are.

---

## 🗣️ Available Voices

| Voice | Locale | Gender |
|-------|--------|--------|
| Aria — Female, US | `en-US` | Female |
| Guy — Male, US | `en-US` | Male |
| Jenny — Female, US | `en-US` | Female |
| Sonia — Female, UK | `en-GB` | Female |
| Ryan — Male, UK | `en-GB` | Male |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Webcam Feed  ─────┐                                    │
│  Image Upload  ────┼──►  Preprocess (resize, enhance)   │
└────────────────────┘              │                       │
                                    ▼                       │
                          ┌──────────────────┐              │
                          │  dHash Check     │◄─────────────┤
                          │  (scene change?)  │              │
                          └────────┬─────────┘              │
                                   │                        │
                    unchanged ─────┘                        │
                    changed ───────►  Florence-2-base        │
                                     (Microsoft)             │
                                     ├─ <CAPTION>            │
                                     ├─ <DETAILED_CAPTION>   │
                                     ├─ <MORE_DETAILED_CAPTION>
                                     ├─ <OCR>               │
                                     └─ <OD> ──► spatial    │
                                                 formatter   │
                                                          │
                                          │               │
                                          ▼               │
                              ┌──────────────────────┐    │
                              │  Edge TTS (async)    │    │
                              │  dedicated thread    │    │
                              └──────────┬───────────┘    │
                                         │                │
                                         ▼                │
                              ┌──────────────────────┐    │
                              │  Gradio Audio        │    │
                              │  (autoplay)          │    │
                              └──────────────────────┘    │
```

**Scene-change detection** uses dHash (difference hash) with a configurable Hamming-distance threshold (`SCENE_THRESHOLD = 0.12`). If the new frame is ≥88% similar to the previous one, the inference is skipped — reducing unnecessary compute and audio fatigue.

**Dedicated TTS event loop** — Edge-TTS runs on its own asyncio event loop in a background thread, so speech generation never blocks the vision model or the UI.

---

## 🚀 Getting Started

### Run on Hugging Face Spaces

1. Click **"Duplicate this Space"** (top-right on the Hugging Face page)
2. Set visibility to **Public** or **Private**
3. Wait for the build (~2–3 minutes on first launch)
4. Open the app — no installation needed!

### Run locally

```bash
# 1. Clone
# git clone <your-repo-url>
cd echolens

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
python app.py
```

The app will be available at `http://localhost:7860`.

> **Note:** The first launch downloads the Florence-2-base model (~460 MB) and performs a background warmup inference. Expect a short delay before the first description.

---

## 📦 Requirements

```
transformers==4.48.0
timm
torch>=2.1.0
torchvision
Pillow>=10.0.0
einops
edge-tts
gradio==5.34.2
numpy
accelerate
```

GPU (CUDA) is used automatically if available; the app falls back to CPU otherwise. On Apple Silicon, MPS is used.

---

## ⚙️ Configuration

All tunable constants live in the `Config` class at the top of `app.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CAPTURE_INTERVAL` | `3.5` | Seconds between realtime captures |
| `SCENE_THRESHOLD` | `0.12` | dHash distance below which a scene is treated as unchanged |
| `MAX_DIM` | `768` | Max image dimension before inference (larger images are downscaled) |
| `TTS_RATE` | `+8%` | Speech speed adjustment for Edge TTS |
| `DEBOUNCE_S` | `0.8` | Minimum seconds between processing frames (prevents rapid-fire) |

---

## ♿ Accessibility Design

EchoLens is built with accessibility as a first-class concern:

| Feature | Implementation |
|---------|---------------|
| **ARIA live regions** | Screen readers auto-announce new descriptions via `aria-live="assertive"` |
| **Keyboard-first** | All core actions reachable without a mouse (D/R/P/Esc) |
| **Font scaling** | Three size levels (A / A+ / A++) via CSS class toggles on `<body>` |
| **High contrast mode** | One-click toggle; increases contrast 1.7× and adds dark borders |
| **Autoplay audio** | Descriptions play immediately; no extra click needed |
| **Large touch targets** | Buttons ≥52 px height; full-width on mobile |
| **Focus indicators** | Visible 3 px focus rings on all interactive elements |
| **Status feedback** | Real-time status bar shows processing state, cache hits, word counts |
| **Screen-reader announcements** | JavaScript pushes button-action feedback to the ARIA live region |

---

## 🧪 Development

### Project structure

```
.
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── LICENSE             # Apache 2.0
```

### Adding a new voice

Add an entry to `VOICE_MAP` in `app.py`:

```python
VOICE_MAP = {
    # ... existing voices ...
    "New Voice — Female, CA": "en-CA-ClaraNeural",
}
```

Find available voices with `edge-tts --list-voices`.

### Adding a new task

1. Add to `TASKS` (human label → Florence-2 token)
2. Add to `TASK_INFO` (tooltip description)
3. Set `MAX_TOKENS` for the token if needed
4. Add formatting logic in `_infer()` if the task needs custom output parsing

---

## 🤖 Model

[Microsoft Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) — a unified vision-language model that handles captioning, OCR, and object detection through task-specific prompt tokens. Runs in `float16` on CUDA and `float32` on CPU/MPS.

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Microsoft Florence-2](https://huggingface.co/microsoft/Florence-2-base) for the vision-language model
- [Edge TTS](https://github.com/rany2/edge-tts) for free, high-quality neural text-to-speech
- [Gradio](https://gradio.app) for the accessible web UI framework
- [Hugging Face](https://huggingface.co) for model hosting and Spaces infrastructure
- [Transformers](https://github.com/huggingface/transformers) for the model inference pipeline
