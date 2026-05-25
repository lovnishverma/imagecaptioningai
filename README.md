---
title: EchoLens 2.0
emoji: 🌍
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: true
license: apache-2.0
short_description: Image to Caption using AI
---

# 👁️ EchoLens 2.0 — Realtime Vision Assistant for Blind & Low-Vision Users

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Gradio](https://img.shields.io/badge/Gradio-5.34.2-orange)](https://gradio.app)
[![Model: Florence-2](https://img.shields.io/badge/Model-Florence--2--base-purple)](https://huggingface.co/microsoft/Florence-2-base)

EchoLens is an accessible, real-time vision assistant that helps blind and low-vision users understand their surroundings through AI-powered image captioning and text-to-speech. Point a webcam at the world, and EchoLens describes what it sees — out loud.

---

## ✨ Features

- **Instant scene description** — Press `D` or click *Describe Now* to hear what the camera sees
- **Realtime mode** — Auto-describes the scene every 3.5 seconds; skips unchanged frames using perceptual hashing
- **Multiple vision tasks** — Quick caption, detailed scene description, OCR (text reading), and object detection with spatial positions (left / center / right)
- **Natural-sounding TTS** — Powered by Microsoft Edge TTS with 5 voice options across US and UK accents
- **Image upload support** — Upload a photo from disk for immediate description
- **Repeat last description** — Press `P` to replay the last audio at any time
- **Accessible UI** — Font size controls (A / A+ / A++), high-contrast toggle, keyboard shortcuts, and ARIA live regions for screen reader compatibility

---

## 🎹 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `D` | Describe what the camera sees right now |
| `R` | Toggle realtime auto-description on/off |
| `P` | Repeat the last description |
| `Esc` | Stop / silence |

---

## 🧠 Vision Tasks

| Task | Description |
|------|-------------|
| **Quick Caption** | Short one-line summary of the scene |
| **Describe Scene** | Detailed multi-sentence description |
| **Read Text (OCR)** | Reads any text visible in the image |
| **Detect Objects** | Lists objects and their positions (left, center, right) |

---

## 🗣️ Available Voices

| Voice | Locale |
|-------|--------|
| Aria (Female, US) | en-US |
| Guy (Male, US) | en-US |
| Jenny (Female, US) | en-US |
| Sonia (Female, UK) | en-GB |
| Ryan (Male, UK) | en-GB |

---

## 🏗️ Architecture

```
Webcam / Upload
      │
      ▼
 dHash check ──── same scene? ──► skip (realtime mode)
      │
      ▼
Florence-2-base (Microsoft)
  ├── <CAPTION>
  ├── <MORE_DETAILED_CAPTION>
  ├── <OCR>
  └── <OD>  ──► spatial formatter (left/center/right)
      │
      ▼
 Edge TTS (async, dedicated event loop)
      │
      ▼
 Gradio Audio (autoplay) + Textbox output
```

**Scene-change detection** uses dHash (difference hash) with a configurable Hamming-distance threshold (`HASH_THRESHOLD = 0.12`), preventing redundant inference when the camera is stationary.

---

## 🚀 Getting Started

### Run on Hugging Face Spaces

Just open the Space — no installation needed.

### Run locally

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/<your-username>/imagecaptioningai
cd imagecaptioningai

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
Pillow>=10.0.0
einops
edge-tts
gradio==5.34.2
```

GPU (CUDA) is used automatically if available; the app falls back to CPU otherwise.

---

## ⚙️ Configuration

Key constants at the top of `app.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CAPTURE_INTERVAL` | `3.5` | Seconds between realtime captures |
| `HASH_THRESHOLD` | `0.12` | dHash distance below which a scene is treated as unchanged |
| `MAX_DIM` | `768` | Maximum image dimension before inference (downscales larger images) |

---

## ♿ Accessibility Design

EchoLens is built with accessibility as a first-class concern:

- **ARIA live regions** — Screen readers announce new descriptions automatically
- **Keyboard-first** — All core actions reachable without a mouse
- **Font scaling** — Three size levels (A / A+ / A++) applied via CSS class toggles
- **High contrast mode** — Increases contrast and brightness for low-vision users
- **Autoplay audio** — Descriptions play immediately without user interaction
- **Large touch targets** — Buttons sized ≥52 px height, full-width on mobile

---

## 🤖 Model

[Microsoft Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) — a compact vision-language model supporting captioning, OCR, and object detection through task-prompt tokens. Runs in `float16` on GPU and `float32` on CPU.

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Microsoft Florence-2](https://huggingface.co/microsoft/Florence-2-base) for the vision model
- [Edge TTS](https://github.com/rany2/edge-tts) for free, high-quality neural TTS
- [Gradio](https://gradio.app) for the accessible web UI framework
- [Hugging Face](https://huggingface.co) for hosting
