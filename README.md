# Pashto & Dari Speech AI — STT + TTS

Production-grade Speech-to-Text and Text-to-Speech pipeline for **Pashto** and **Dari (Afghan Persian)** — two low-resource languages with extremely scarce open-source tooling. Models are fine-tuned from Foundation Models on large-scale datasets, optimized via quantization, and deployed as auto-scaling REST APIs on Modal Cloud.

---

## Models & Training Results

### Dari STT — Whisper Large-v3 + LoRA

| Detail | Value |
|---|---|
| Base model | OpenAI Whisper Large-v3 (1.55B params) |
| Dataset | ~85K Dari audio clips (~189 hours) |
| Fine-tuning | LoRA rank 64, 3.6% trainable params, SpecAugment augmentation |
| Training | 32.6 GPU hours |
| **WER** | **12.29%** (1,000-sample test set) |
| **CER** | **2.62%** (39.9% perfect transcriptions) |
| Inference | CTranslate2 INT8 quantization → 1.5 GB model |
| Warm latency | ~2.31s on T4 GPU (Modal Cloud) |

### Pashto STT — Whisper Large-v3 + LoRA

| Detail | Value |
|---|---|
| Base model | OpenAI Whisper Large-v3 (1.55B params) |
| Fine-tuning | LoRA rank 32, native Pashto language code (`ps`) |
| Inference | CTranslate2 INT8 quantization → 1.5 GB VRAM at runtime |
| **Warm latency** | **~0.7s/request** |
| Audio support | Up to 90-second clips via VAD-based auto-chunking |

### Dari TTS — MB-iSTFT-VITS2 (trained from scratch)

| Detail | Value |
|---|---|
| Architecture | Multi-Band iSTFT VITS2 GAN (~40M params) |
| Dataset | 41,372 Dari speech clips at 22 kHz |
| Training | 120,000 steps from scratch |
| Custom preprocessing | Dari text cleaner: normalization, number-to-word, Unicode/Perso-Arabic script |
| Checkpoint | 441 MB |
| **Warm latency** | **1.88s on T4 GPU** |

### Pashto TTS — VITS2 (trained from scratch)

| Detail | Value |
|---|---|
| Architecture | VITS2 GAN-based TTS, 22 kHz output |
| Checkpoint | 480 MB |
| **Warm latency** | **1.61s on T4 GPU** |
| Demo | Local Gradio + Modal REST API |

---

## Infrastructure

All 4 models are deployed as independent auto-scaling containers on Modal Cloud:

```
Client Request
      ↓
REST API Layer (HMAC-based auth, constant-time key verification)
      ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Dari STT    │  │ Pashto STT   │  │  Dari TTS    │  │ Pashto TTS   │
│ Whisper INT8 │  │ Whisper INT8 │  │ MB-iSTFT     │  │ VITS2        │
│ T4 GPU       │  │ T4 GPU       │  │ VITS2  T4    │  │ T4 GPU       │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

- Per-language GPU assignment with cold-start isolation per container
- Dual-workspace deployment (~3.8 GB model weights per workspace)
- End-to-end cold/warm latency benchmarked across all 4 endpoints
- Browser UI via Node.js proxy: API key never exposed to client, per-session rate limiting

---

## Key Engineering Details

**Module isolation** — Both TTS packages expose identical top-level Python module names. A `_tts_env()` context manager adds the package dir to `sys.path`, captures all references, then scrubs matching keys from `sys.modules` before the next package loads, allowing both to coexist in one FastAPI process.

**Cython → NumPy fallback** — VITS2's monotonic alignment search is a Cython C-extension. A pure-NumPy fallback (`monotonic_align/__init__.py`) auto-selects when the compiled extension is absent, unblocking Windows development with bit-for-bit identical output.

**Cold-start mitigation** — Modal containers cold-start in ~40s. Two strategies: retry on 503 after 15s (transparent to user); parallel TTS warm-up after every STT request so the next TTS call hits a warm container.

---

## Tech Stack

| Category | Tools |
|---|---|
| Model training | PyTorch, CUDA, Hugging Face Transformers |
| Fine-tuning | LoRA / PEFT, SpecAugment |
| Inference optimization | CTranslate2, faster-whisper, INT8 quantization |
| TTS architecture | VITS2, MB-iSTFT-VITS2 |
| Deployment | Modal Cloud (serverless GPU, auto-scaling) |
| Web proxy | Node.js + Express, express-session, multer, ffmpeg |
| Evaluation | WER, CER, per-sample accuracy metrics |

---

## Project Structure

```
.
├── api.py                          # FastAPI local backend (STT + TTS)
├── requirements_api.txt
│
├── Finalized_VITS2_Pashto_TTS/    # Pashto MB-iSTFT-VITS2 source
├── dari_tts_deploy_v1/            # Dari MB-iSTFT-VITS2 source
├── pashto_ct2_int8/               # Whisper CT2 model config (weights excluded)
├── dari_ct2_int8/                 # Whisper CT2 model config (weights excluded)
│
└── voice-app/                     # Node.js web application
    ├── server.js
    ├── public/                    # RTL-aware browser UI (MediaRecorder)
    └── src/
        ├── modalClient.js         # cold-start retry + warmup
        ├── rateLimiter.js         # per-session + global limits
        └── audioUtils.js          # ffmpeg 16kHz conversion
```

---

## Local Setup

**Python backend** (model weights required separately):
```bash
pip install -r requirements_api.txt
# Place weights: pashto_ct2_int8/model.bin, dari_ct2_int8/model.bin, TTS dirs
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Node.js web app**:
```bash
cd voice-app
cp .env.example .env   # add Modal API key
npm install && npm start
# → http://localhost:3000
```

Requires `ffmpeg` in PATH (`winget install ffmpeg` on Windows).

---

## Related Projects

- [whatsapp-ai-agent](https://github.com/Tameemkharbey/whatsapp-ai-agent) — WhatsApp bot for Pashto/Dari banking support powered by these Modal endpoints

---

## License

MIT. Model weights excluded — subject to respective training data licenses.
