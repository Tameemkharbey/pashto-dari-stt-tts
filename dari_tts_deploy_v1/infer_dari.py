"""
Dari MB-iSTFT-VITS2 single-speaker inference.

Usage:
  python infer_dari.py --checkpoint logs/dari_model/G_50000.pth --out samples
"""
import argparse
import os
import sys
import time

import torch
from scipy.io.wavfile import write

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


# 10 diverse Dari sentences from the held-out val set — short, medium, long.
SENTENCES = [
    "حق گوی اگر چه تلخ استه.",
    "درخت آلبالو فکر موکد",
    "۱۵. دقت آماری",
    "بلبل هزار داستان واری میخوانه.",
    "تنوع در بازارهای مصرفی",
    "تأثیر بر معماری اروپایی و بیزانسی در قرن وسطی.",
    "اصول اخلاق فردی صداقت، انصاف، احترام به دیگران، تواضع و شایستگی.",
    "حفظ نظم و عدالت اجتماعی با ایجاد دیوان‌ها و نهادهای قضایی.",
    "در این ادبیات، عاطفه و احساس در کنار آگاهی و تفکر، دو بال اصلی پرواز اندیشه‌استه",
    "بازار محلی میتنه مرکز آموزش و فرهنگ بشه، زیرا مردم در حین خرید با رسوم و محصولات محلی آشنا موشه",
]


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dari_single_speaker.json")
    parser.add_argument("--checkpoint", required=True, help="Path to G_*.pth")
    parser.add_argument("--out", default="samples", help="Output directory")
    parser.add_argument("--noise_scale", type=float, default=0.667)
    parser.add_argument("--noise_scale_w", type=float, default=0.8)
    parser.add_argument("--length_scale", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    hps = utils.get_hparams_from_file(args.config)

    if getattr(hps.model, "use_mel_posterior_encoder", False) or \
       getattr(hps.data, "use_mel_posterior_encoder", False):
        print("[infer] Using mel posterior encoder (VITS2)")
        posterior_channels = 80
        hps.data.use_mel_posterior_encoder = True
    else:
        print("[infer] Using lin posterior encoder (VITS1)")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    net_g.eval()

    print(f"[infer] Loading checkpoint: {args.checkpoint}")
    utils.load_checkpoint(args.checkpoint, net_g, None)

    print(f"[infer] Generating {len(SENTENCES)} samples -> {args.out}/")
    total_audio_sec = 0.0
    total_rtf_time = 0.0

    for i, text in enumerate(SENTENCES, 1):
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.to(device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

            t0 = time.time()
            audio = net_g.infer(
                x_tst,
                x_tst_lengths,
                noise_scale=args.noise_scale,
                noise_scale_w=args.noise_scale_w,
                length_scale=args.length_scale,
            )[0][0, 0].data.cpu().float().numpy()
            elapsed = time.time() - t0

        sr = hps.data.sampling_rate
        dur = len(audio) / sr
        rtf = elapsed / dur if dur > 0 else 0.0
        total_audio_sec += dur
        total_rtf_time += elapsed

        out_path = os.path.join(args.out, f"dari_sample_{i:02d}.wav")
        write(out_path, sr, audio)
        print(f"  [{i:02d}] {dur:5.2f}s  RTF={rtf:.3f}  {out_path}")
        print(f"        {text}")

    overall_rtf = total_rtf_time / total_audio_sec if total_audio_sec > 0 else 0.0
    print(f"\n[infer] Done. Total audio={total_audio_sec:.1f}s  overall RTF={overall_rtf:.3f}")
    print(f"[infer] Listen to {args.out}/dari_sample_*.wav")


if __name__ == "__main__":
    main()
