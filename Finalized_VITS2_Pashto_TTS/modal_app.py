"""
Modal deployment for Finalized VITS2 Pashto TTS — API only (no UI).

Deploy:
    modal deploy modal_app.py

Endpoints (after deploy):
    POST  https://<username>--pashto-tts-vits2-pashtotts-synthesize.modal.run
          Body: {"text": "...", "noise_scale": 0.4, "noise_scale_w": 0.8, "length_scale": 1.0}
          Returns: audio/wav bytes

    GET   https://<username>--pashto-tts-vits2-pashtotts-health.modal.run
          Returns: {"status": "ok"}
"""

import os
from pathlib import Path

import modal

APP_NAME = "pashto-tts-vits2"
GPU_TYPE = "T4"
LOCAL_MODEL_DIR = Path(__file__).parent
REMOTE_MODEL_DIR = "/model_root"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "numpy==1.24.4",
        "scipy==1.11.4",
        "librosa==0.10.1",
        "numba==0.58.1",
        "Unidecode==1.3.7",
        "phonemizer==3.2.1",
        "tensorboardX",
        "tqdm",
        "Cython",
        "fastapi==0.115.6",
        "pydantic==2.9.2",
    )
    .add_local_dir(
        LOCAL_MODEL_DIR.as_posix(),
        REMOTE_MODEL_DIR,
        ignore=[
            "__pycache__",
            "*.pyc",
            "modal_app.py",
            "samples",
            "filelists",
            "scripts",
        ],
    )
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=GPU_TYPE,
    scaledown_window=300,
    max_containers=2,
)
@modal.concurrent(max_inputs=4)
class PashtoTTS:
    @modal.enter()
    def load_model(self):
        import sys
        import json
        import torch

        sys.path.insert(0, REMOTE_MODEL_DIR)

        from text.symbols import symbols
        from text import text_to_sequence
        from models import SynthesizerTrn
        import commons

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading VITS2 Pashto model on {self.device}...")

        with open(os.path.join(REMOTE_MODEL_DIR, "configs", "pashto.json")) as f:
            self.hps = json.load(f)

        self.net_g = SynthesizerTrn(
            len(symbols),
            80,
            self.hps["train"]["segment_size"] // self.hps["data"]["hop_length"],
            n_speakers=self.hps["data"]["n_speakers"],
            **self.hps["model"],
        ).to(self.device)

        ckpt_path = os.path.join(REMOTE_MODEL_DIR, "model", "G_314000.pth")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.net_g.load_state_dict(checkpoint["model"])
        self.net_g.eval()

        self._text_to_sequence = text_to_sequence
        self._commons = commons
        self._torch = torch

        print("Model loaded and ready!")

    def _run(self, text: str, noise_scale: float, noise_scale_w: float, length_scale: float):
        import io
        import numpy as np
        from scipy.io import wavfile

        torch = self._torch
        seq = self._text_to_sequence(text.strip(), ["pashto_cleaners"])
        if self.hps["data"]["add_blank"]:
            seq = self._commons.intersperse(seq, 0)

        x = torch.LongTensor(seq).unsqueeze(0).to(self.device)
        x_len = torch.LongTensor([len(seq)]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)

        with torch.no_grad():
            audio = self.net_g.infer(
                x, x_len, sid=sid,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )

        audio_np = audio[0][0].data.cpu().float().numpy().squeeze()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (audio_np * 32767.0).astype(np.int16)

        buf = io.BytesIO()
        wavfile.write(buf, 22050, pcm16)
        return buf.getvalue()

    @modal.fastapi_endpoint(method="POST", docs=True)
    def synthesize(self, item: dict):
        from fastapi import Response, HTTPException

        text = (item or {}).get("text", "")
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Field 'text' is required and non-empty.")

        noise_scale = float((item or {}).get("noise_scale", 0.4))
        noise_scale_w = float((item or {}).get("noise_scale_w", 0.8))
        length_scale = float((item or {}).get("length_scale", 1.0))

        wav_bytes = self._run(text, noise_scale, noise_scale_w, length_scale)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="pashto.wav"'},
        )

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ok", "model": "VITS2", "device": self.device}


