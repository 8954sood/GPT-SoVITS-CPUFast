"""Original-rate log-mel/STFT supervision for the V2ProPlus decoder student."""

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F


class VoiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mels = nn.ModuleList(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=32000,
                n_fft=size,
                hop_length=size // 4,
                n_mels=bins,
                power=1.0,
                norm="slaney",
                mel_scale="slaney",
            )
            for size, bins in ((512, 64), (1024, 100), (2048, 128))
        )
        for size in (512, 1024, 2048):
            self.register_buffer(f"window_{size}", torch.hann_window(size))

    def forward(self, audio, target):
        if audio.shape != target.shape:
            raise ValueError("Equal original-rate waveform shapes required")
        audio, target = audio.flatten(0, 1), target.flatten(0, 1)
        mel = torch.stack(
            [
                F.l1_loss(
                    transform(audio).clamp_min(1e-5).log(),
                    transform(target).clamp_min(1e-5).log(),
                )
                for transform in self.mels
            ]
        ).mean()
        stft = []
        for size in (512, 1024, 2048):
            options = {
                "n_fft": size,
                "hop_length": size // 4,
                "window": getattr(self, f"window_{size}"),
                "return_complex": True,
            }
            a = torch.stft(audio, **options).abs().clamp_min(1e-5).log()
            b = torch.stft(target, **options).abs().clamp_min(1e-5).log()
            stft.append(F.l1_loss(a, b))
        return mel, torch.stack(stft).mean()
