# Finalized VITS2 Pashto TTS

Pashto Text-to-Speech model using MB-iSTFT-VITS2 architecture.

## Quick Start

```bash
cd Finalized_VITS2_Pashto_TTS
python3 app.py
# Opens Gradio at http://localhost:7860
```

## Folder Structure

```
Finalized_VITS2_Pashto_TTS/
│
├── app.py                  # Gradio web interface (MAIN ENTRY POINT)
├── README.md               # This file
├── requirements.txt        # Python dependencies
│
├── model/                  # Trained model checkpoint
│   └── G_314000.pth        # Trained generator checkpoint
│
├── configs/                # Model configuration
│   └── pashto.json         # Model configuration
│
├── text/                   # Pashto text processing
│   ├── __init__.py         # text_to_sequence() function
│   ├── symbols.py          # Pashto character vocabulary
│   └── cleaners.py         # pashto_cleaners
│
├── samples/                # Pre-generated audio samples for reference
│   ├── samples.txt         # Sample descriptions
│   └── *.wav               # 12 listening test files
│
├── scripts/                # Training & evaluation (if needed later)
│   ├── train_ms.py
│   ├── evaluate_all.py
│   ├── inference.py
│   └── preprocess.py
│
├── # Core model modules (required by app.py)
├── models.py               # SynthesizerTrn architecture
├── modules.py              # Neural network modules
├── attentions.py           # Attention layers
├── commons.py              # Utilities
├── mel_processing.py       # Mel spectrogram processing
├── transforms.py           # Audio transforms
├── utils.py                # General utilities
├── stft.py                 # Short-time Fourier transform
├── stft_loss.py            # STFT loss computation
├── pqmf.py                 # Pseudo-QMF filter bank
├── losses.py               # Loss functions
├── data_utils.py           # Data loading
├── S_monotonic_align.py    # Duration alignment
└── monotonic_align/        # Monotonic alignment module
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | MB-iSTFT-VITS2 (Multi-Band iSTFT) |
| Parameters | ~40M |
| Sample Rate | 22050 Hz |
| Recommended noise_scale | 0.4 |

## Pashto Character Set (59 chars)

The model supports standard Arabic-based characters plus Pashto-specific: ښ, ځ, څ, ړ, ږ, ګ, ڼ, ۍ, ې, etc.

## Requirements

- Python 3.10+
- PyTorch with CUDA
- GPU with 4GB+ VRAM (inference only)
- See requirements.txt for full list
