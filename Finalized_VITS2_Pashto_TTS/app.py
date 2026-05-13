"""
Pashto TTS — Gradio Demo
=========================
Run: python3 app.py
Opens at: http://localhost:7860
"""

import os
import torch
import json
import numpy as np
import warnings
import gradio as gr
from text import text_to_sequence
from text.symbols import symbols
from models import SynthesizerTrn
import commons

warnings.filterwarnings("ignore")

# Load model once at startup
print("Loading model...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "configs", "pashto.json")) as f:
    hps = json.load(f)

net_g = SynthesizerTrn(
    len(symbols), 80,
    hps["train"]["segment_size"] // hps["data"]["hop_length"],
    n_speakers=hps["data"]["n_speakers"],
    **hps["model"],
).cuda()

checkpoint = torch.load(os.path.join(BASE_DIR, "model", "G_314000.pth"), map_location="cpu")
net_g.load_state_dict(checkpoint["model"])
net_g.eval()
print("Model loaded! Starting Gradio...")


def synthesize(text, noise_scale, noise_scale_w, length_scale):
    if not text or not text.strip():
        return None

    seq = text_to_sequence(text.strip(), ["pashto_cleaners"])
    if hps["data"]["add_blank"]:
        seq = commons.intersperse(seq, 0)

    x = torch.LongTensor(seq).unsqueeze(0).cuda()
    x_len = torch.LongTensor([len(seq)]).cuda()
    sid = torch.LongTensor([0]).cuda()

    with torch.no_grad():
        audio = net_g.infer(
            x, x_len, sid=sid,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )

    audio_np = audio[0][0].data.cpu().float().numpy().squeeze()
    audio_np = np.clip(audio_np, -1.0, 1.0)

    return (22050, audio_np)


# Example sentences
examples = [
    ["سلام"],
    ["مننه"],
    ["سلام، زه ستاسو سره مرسته کولی شم"],
    ["افغانستان یو ښکلی هېواد دی"],
    ["زه هره ورځ مکتب ته ځم"],
    ["دا زموږ د پښتو ژبې تاریخ دی"],
    ["سوله او امنیت ډېر مهم دي"],
    ["نن ورځ هوا ډېره ښه ده"],
    ["زه پښتون یم او پښتو خبرې کوم"],
    ["زموږ د پښتو ژبې تاریخ ډېر پخوانی دی او دا ژبه د نړۍ په مختلفو برخو کې ویل کیږي"],
]

demo = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Textbox(
            label="Pashto Text",
            placeholder="پښتو متن دلته ولیکئ...",
            lines=3,
            rtl=True,
        ),
        gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Noise Scale (expressiveness)"),
        gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Noise Scale W (duration variation)"),
        gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Length Scale (speed: <1 faster, >1 slower)"),
    ],
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    title="Pashto TTS - VITS2",
    description="Pashto text-to-speech model. Type any Pashto text and click Submit.",
    examples=examples,
    flagging_mode="never",
)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
