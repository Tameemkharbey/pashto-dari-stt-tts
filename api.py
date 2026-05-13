"""
Pashto & Dari STT / TTS — FastAPI Backend
==========================================

Endpoints
---------
GET  /health
POST /stt?language=pashto|dari    body: multipart audio file (WAV/MP3/OGG/FLAC/M4A)
POST /tts?language=pashto|dari    body: form  text, noise_scale, noise_scale_w, length_scale

Run
---
pip install -r requirements_api.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from scipy.io.wavfile import write as _wav_write

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
PASHTO_TTS = ROOT / "Finalized_VITS2_Pashto_TTS"
DARI_TTS   = ROOT / "dari_tts_deploy_v1"
PASHTO_STT = ROOT / "pashto_ct2_int8"
DARI_STT   = ROOT / "dari_ct2_int8"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Module-isolation helper ───────────────────────────────────────────────────
# Both TTS packages expose top-level modules with identical names (models,
# commons, text, …).  We load each package inside this context manager, which
# adds its directory to sys.path and scrubs those module names from sys.modules
# afterwards so the second package can import its own copies cleanly.
_SHARED_NAMES = {
    "models", "commons", "utils", "modules", "attentions",
    "mel_processing", "transforms", "stft", "pqmf", "losses",
    "data_utils", "S_monotonic_align", "monotonic_align",
    "text", "text.symbols", "text.cleaners",
}


@contextmanager
def _tts_env(directory: Path):
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(directory))
        except ValueError:
            pass
        for key in list(sys.modules):
            if key in _SHARED_NAMES or any(
                key.startswith(m + ".") for m in _SHARED_NAMES
            ):
                del sys.modules[key]


# ── Load Pashto TTS ───────────────────────────────────────────────────────────
print("[api] Loading Pashto TTS …", flush=True)

with _tts_env(PASHTO_TTS):
    with open(PASHTO_TTS / "configs" / "pashto.json") as _f:
        _ps_cfg = json.load(_f)
    from text.symbols import symbols as _ps_sym
    from text import text_to_sequence as _ps_t2s
    import commons as _ps_commons
    from models import SynthesizerTrn as _PsTrn
    import utils as _ps_utils
    _ps_load_ckpt = _ps_utils.load_checkpoint

_ps_model: torch.nn.Module = _PsTrn(
    len(_ps_sym),
    80,
    _ps_cfg["train"]["segment_size"] // _ps_cfg["data"]["hop_length"],
    n_speakers=_ps_cfg["data"]["n_speakers"],
    **_ps_cfg["model"],
).to(DEVICE)
_ps_load_ckpt(str(PASHTO_TTS / "model" / "G_314000.pth"), _ps_model, None)
_ps_model.eval()
print("[api] Pashto TTS ready.", flush=True)

# ── Load Dari TTS ─────────────────────────────────────────────────────────────
print("[api] Loading Dari TTS …", flush=True)

with _tts_env(DARI_TTS):
    with open(DARI_TTS / "configs" / "dari_single_speaker.json") as _f:
        _dr_cfg = json.load(_f)
    from text.symbols import symbols as _dr_sym
    from text import text_to_sequence as _dr_t2s
    import commons as _dr_commons
    from models import SynthesizerTrn as _DrTrn
    import utils as _dr_utils
    _dr_load_ckpt = _dr_utils.load_checkpoint

_dr_model: torch.nn.Module = _DrTrn(
    len(_dr_sym),
    80,
    _dr_cfg["train"]["segment_size"] // _dr_cfg["data"]["hop_length"],
    n_speakers=_dr_cfg["data"]["n_speakers"],
    **_dr_cfg["model"],
).to(DEVICE)
_dr_load_ckpt(str(DARI_TTS / "G_120000.pth"), _dr_model, None)
_dr_model.eval()
print("[api] Dari TTS ready.", flush=True)

# ── Load STT models ───────────────────────────────────────────────────────────
print("[api] Loading STT models …", flush=True)
from faster_whisper import WhisperModel

_ps_stt = WhisperModel(str(PASHTO_STT), device="cpu", compute_type="int8")
_dr_stt = WhisperModel(str(DARI_STT),   device="cpu", compute_type="int8")
print("[api] STT models ready.", flush=True)


# ── Inference helpers ─────────────────────────────────────────────────────────
def _audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    _wav_write(buf, sr, pcm)
    return buf.getvalue()


def _synth_pashto(
    text: str,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
) -> bytes:
    seq = _ps_t2s(text.strip(), _ps_cfg["data"]["text_cleaners"])
    if _ps_cfg["data"]["add_blank"]:
        seq = _ps_commons.intersperse(seq, 0)
    x     = torch.LongTensor(seq).unsqueeze(0).to(DEVICE)
    x_len = torch.LongTensor([len(seq)]).to(DEVICE)
    sid   = torch.LongTensor([0]).to(DEVICE)
    with torch.no_grad():
        audio = (
            _ps_model.infer(
                x, x_len, sid=sid,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu().float().numpy()
        )
    return _audio_to_wav_bytes(audio, _ps_cfg["data"]["sampling_rate"])


def _synth_dari(
    text: str,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
) -> bytes:
    seq = _dr_t2s(text.strip(), _dr_cfg["data"]["text_cleaners"])
    if _dr_cfg["data"]["add_blank"]:
        seq = _dr_commons.intersperse(seq, 0)
    x     = torch.LongTensor(seq).unsqueeze(0).to(DEVICE)
    x_len = torch.LongTensor([len(seq)]).to(DEVICE)
    with torch.no_grad():
        audio = (
            _dr_model.infer(
                x, x_len,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu().float().numpy()
        )
    return _audio_to_wav_bytes(audio, _dr_cfg["data"]["sampling_rate"])


def _transcribe(audio_bytes: bytes, language: str) -> dict:
    lang_code = "ps" if language == "pashto" else "fa"
    stt_model = _ps_stt if language == "pashto" else _dr_stt
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        segments, info = stt_model.transcribe(
            tmp_path, language=lang_code, beam_size=5
        )
        text = "".join(seg.text for seg in segments).strip()
    finally:
        os.unlink(tmp_path)
    return {
        "text": text,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_seconds": round(info.duration, 2),
    }


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pashto & Dari STT/TTS API",
    version="1.0.0",
    description="Speech-to-text and text-to-speech for Pashto and Dari (Afghan Persian).",
)


@app.get("/health", summary="Model health check")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "models": {
            "pashto_tts": "MB-iSTFT-VITS2, 22050 Hz",
            "dari_tts":   "MB-iSTFT-VITS2, 22050 Hz",
            "pashto_stt": "Whisper CT2 INT8",
            "dari_stt":   "Whisper CT2 INT8",
        },
    }


@app.post(
    "/stt",
    summary="Speech → Text",
    response_description="Transcription result",
)
async def speech_to_text(
    language: Literal["pashto", "dari"] = Query(
        ..., description="Language of the audio"
    ),
    audio: UploadFile = File(
        ..., description="Audio file (WAV, MP3, OGG, FLAC, M4A)"
    ),
):
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        return _transcribe(data, language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/tts",
    summary="Text → Speech",
    response_description="WAV audio file",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
)
async def text_to_speech(
    language: Literal["pashto", "dari"] = Query(
        ..., description="Language to synthesize"
    ),
    text: str = Form(..., description="Input text"),
    noise_scale: float = Form(
        0.4, ge=0.0, le=1.0,
        description="Expressiveness (0 = flat, 1 = expressive). Default 0.4.",
    ),
    noise_scale_w: float = Form(
        0.8, ge=0.0, le=1.0,
        description="Duration variation. Default 0.8.",
    ),
    length_scale: float = Form(
        1.0, ge=0.5, le=2.0,
        description="Speed: <1 faster, >1 slower. Default 1.0.",
    ),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    try:
        if language == "pashto":
            wav = _synth_pashto(text, noise_scale, noise_scale_w, length_scale)
        else:
            wav = _synth_dari(text, noise_scale, noise_scale_w, length_scale)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{language}_tts.wav"'
        },
    )
