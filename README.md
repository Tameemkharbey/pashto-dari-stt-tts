# Pashto & Dari — Speech-to-Text and Text-to-Speech System

End-to-end STT + TTS pipeline for **Pashto** and **Dari (Afghan Persian)** — two low-resource languages with very limited open-source tooling. Models are deployed serverlessly on [Modal](https://modal.com) and served through a Node.js proxy with rate limiting, session tracking, and a real-time browser UI.

---

## Architecture

```
Browser (MediaRecorder / file upload)
        │
        ▼
┌─────────────────────────────┐
│  Node.js Proxy  (port 3000) │
│  ├─ express-session         │  ← per-user rate limiting
│  ├─ multer (5 MB audio)     │  ← multipart upload
│  ├─ ffmpeg (16 kHz convert) │  ← normalise browser WebM → WAV
│  └─ modalClient.js          │  ← cold-start retry logic
└────────────┬────────────────┘
             │  X-API-Key (server-side only, never exposed to client)
             ▼
┌────────────────────────────────────────┐
│         Modal Serverless Cloud         │
│                                        │
│  ┌─────────────┐   ┌───────────────┐  │
│  │  Pashto STT │   │   Dari STT    │  │
│  │ Whisper INT8│   │ Whisper INT8  │  │
│  │ (CT2 beam=5)│   │ lang="fa"     │  │
│  └─────────────┘   └───────────────┘  │
│                                        │
│  ┌─────────────┐   ┌───────────────┐  │
│  │  Pashto TTS │   │   Dari TTS    │  │
│  │ MB-iSTFT    │   │ MB-iSTFT      │  │
│  │ VITS2       │   │ VITS2         │  │
│  │ 22050 Hz    │   │ 22050 Hz      │  │
│  └─────────────┘   └───────────────┘  │
└────────────────────────────────────────┘
```

---

## Models

| Component | Model | Detail |
|---|---|---|
| **Pashto STT** | Whisper (fine-tuned) | Quantized INT8 via CTranslate2, `language="ps"` |
| **Dari STT** | Whisper (fine-tuned) | Quantized INT8 via CTranslate2, `language="fa"` |
| **Pashto TTS** | MB-iSTFT-VITS2 | Single speaker, 22050 Hz |
| **Dari TTS** | MB-iSTFT-VITS2 | Single speaker, 22050 Hz |

Model weights are not included in this repo (each ~450–1500 MB). Download them separately and place them as described in [Local Setup](#local-setup).

---

## Key Engineering Decisions

### 1. Module namespace isolation
Both TTS packages (Pashto and Dari) expose identical top-level Python module names (`models`, `commons`, `text`, `utils`, …). Loading both in the same process causes silent import collisions.

Solved with a `_tts_env()` context manager that adds the package directory to `sys.path`, captures all needed references, then scrubs matching keys from `sys.modules` before the next package loads — allowing both to coexist cleanly in one FastAPI process.

### 2. Cython → NumPy fallback (Windows dev)
VITS2's monotonic alignment search is a Cython C-extension that requires MSVC to compile on Windows. Rather than blocking local development, `monotonic_align/__init__.py` implements the same algorithm in pure NumPy and auto-selects it when the compiled extension is absent. The output is bit-for-bit identical — Cython is only faster, not more accurate.

```python
try:
    from .monotonic_align.core import maximum_path_c as _maximum_path_c
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False   # NumPy fallback used automatically
```

### 3. Modal cold-start mitigation
Modal serverless containers cold-start in ~40 s. Two strategies are applied:
- **Retry on 503**: wait 15 s, retry once — transparent to the user.
- **Parallel TTS warm-up**: after every STT request, a fire-and-forget TTS call to the same language warms that container so the next TTS request is fast.

### 4. API key never reaches the browser
The Node.js server is a strict proxy. The Modal API key lives in `.env` server-side only. The browser calls `/api/stt` and `/api/tts` — it never sees the Modal endpoint URLs or credentials.

### 5. Rate limiting (in-memory, per session)
```
3 requests / minute  per user
10 requests / hour   per user
25 requests / day    per user
40 requests / hour   per IP
1 concurrent request per user  → 409 if violated
400 requests / day   global    → 503
```

---

## Project Structure

```
.
├── api.py                          # FastAPI local backend (STT + TTS)
├── requirements_api.txt            # Python dependencies
│
├── Finalized_VITS2_Pashto_TTS/    # Pashto MB-iSTFT-VITS2 source
│   ├── models.py
│   ├── monotonic_align/
│   │   └── __init__.py            # Cython + NumPy fallback
│   ├── text/
│   └── configs/pashto.json
│
├── dari_tts_deploy_v1/            # Dari MB-iSTFT-VITS2 source
│   ├── models.py
│   ├── monotonic_align/
│   └── configs/dari_single_speaker.json
│
├── pashto_ct2_int8/               # Whisper CT2 model config (weights excluded)
│   ├── config.json
│   ├── tokenizer.json
│   └── vocabulary.json
│
├── dari_ct2_int8/                 # Whisper CT2 model config (weights excluded)
│   ├── config.json
│   ├── tokenizer.json
│   └── vocabulary.json
│
└── voice-app/                     # Node.js web application
    ├── server.js
    ├── package.json
    ├── .env.example
    ├── public/
    │   ├── index.html             # RTL-aware two-panel UI
    │   ├── style.css
    │   └── app.js                 # MediaRecorder + fetch
    └── src/
        ├── config.js
        ├── routes.js              # /api/stt  /api/tts
        ├── modalClient.js         # cold-start retry + warmup
        ├── audioUtils.js          # ffmpeg 16kHz conversion
        ├── rateLimiter.js
        └── logger.js
```

---

## Local Setup

### Python backend (optional — models required)

```bash
pip install -r requirements_api.txt

# Place model weights in their respective directories:
# pashto_ct2_int8/model.bin
# dari_ct2_int8/model.bin
# Finalized_VITS2_Pashto_TTS/model/
# dari_tts_deploy_v1/

uvicorn api:app --host 0.0.0.0 --port 8000
```

### Node.js web app (Modal API key required)

```bash
cd voice-app
cp .env.example .env
# Fill in your Modal API key in .env

npm install
npm start
# → http://localhost:3000
```

Requires **ffmpeg** in PATH for browser audio conversion (`winget install ffmpeg` on Windows).

---

## Tech Stack

**ML / Python**
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 INT8 Whisper inference
- [MB-iSTFT-VITS2](https://github.com/MasayaKawamura/MB-iSTFT-VITS) — multi-band TTS synthesis
- FastAPI + Uvicorn — local inference server

**MLOps / Deployment**
- [Modal](https://modal.com) — serverless GPU/CPU inference with auto-scaling
- ffmpeg — audio normalisation (any format → WAV 16 kHz mono)

**Web**
- Node.js + Express — proxy server with session management
- Browser MediaRecorder API — in-browser microphone recording
- RTL CSS layout — native right-to-left support for Arabic-script languages

---

## Languages

**Pashto** (`ps`) and **Dari** (`fa`) are the two official languages of Afghanistan. Both use Arabic script written right-to-left. Pashto has approximately 40–60 million speakers; Dari is the lingua franca across Afghanistan with ~25 million native speakers.

Open-source STT/TTS tooling for these languages is extremely scarce — most commercial systems do not support them. This project demonstrates that production-quality speech AI is achievable for low-resource languages with fine-tuned Whisper and VITS2 architectures.

---

## License

Code in this repository is MIT licensed. Model weights are excluded and subject to their respective training data licenses.
