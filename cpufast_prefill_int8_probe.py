"""Isolated approximate W8A8 prompt FFNs; learned parameters stay immutable.

Default engines and original decode stay unchanged. Quality approval is required
before measuring full TTS on N1. Primitive equality is to the explicit quantized
reference, never to original FP32 arithmetic.
"""

import hashlib
import json
from pathlib import Path

import torch

from cpufast_prefill_probe import PackedFFNPrefillProxy


def load_library(path):
    root = Path(__file__).resolve().parent
    path = Path(path)
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if (
        manifest["torch_version"] != torch.__version__
        or manifest["sources"].keys() != {"native/cpufast_prefill_int8.cpp"}
        or digest(path) != manifest["library_sha256"]
        or any(
            digest(root / name) != expected
            for name, expected in manifest["sources"].items()
        )
    ):
        raise RuntimeError("W8A8 prefill library/source mismatch")
    torch.ops.load_library(str(path))


class Int8PrefillLinear:
    def __init__(self, weight_t, bias, *, relu=False):
        self.weight_t, self.bias, self.relu = weight_t, bias, relu
        self._key = self._packed = self._scale = self._quantized = None

    def __call__(self, x):
        if (
            torch.is_grad_enabled()
            or torch.is_autocast_enabled("cpu")
            or x.ndim != 2
            or x.shape[0] <= 1
        ):
            raise ValueError(
                "W8A8 prefill requires multi-row inference; no decode/GEMV"
            )
        w = self.weight_t
        if w.is_inference():
            raise ValueError("Versioned original parameter required")
        key = (id(w), w._version, w.data_ptr(), w.dtype, w.device)
        if key != self._key:
            if (
                w.device.type != "cpu"
                or w.dtype != torch.float32
                or w.ndim != 2
                or not w.is_contiguous()
                or not torch.isfinite(w).all()
                or not torch.equal(w, w.half().float())
            ):
                raise ValueError(
                    "W8A8 source must be immutable CPU FP32 exact-half copy"
                )
            # Same per-output weight grid as existing W8A8 decode packq.
            scale = (w.abs().amax(dim=0) / 127).clamp_min(1e-12).contiguous()
            quantized = (
                (w.T / scale[:, None])
                .round()
                .clamp(-127, 127)
                .to(torch.int8)
                .contiguous()
            )
            packed = torch.ops.cpufast_cpp.prefill_int8_pack_rhs(quantized)
            self._packed, self._scale, self._quantized, self._key = (
                packed,
                scale,
                quantized,
                key,
            )
        return torch.ops.cpufast_cpp.prefill_int8_linear(
            x, self._packed, self._scale, self.bias, self.weight_t.shape[0], self.relu
        )


class Int8FFNPrefillProxy(PackedFFNPrefillProxy):
    """Reuse scoped restoration/decode delegation; supply only new FFN arithmetic."""

    def __init__(self, original):
        self.original = original
        self.busy = False
        self.prefill_calls = 0
        self.forwarders, self.prefill_linears, self.audit = [], [], []
        for index, cached in enumerate(original.blocks):
            mlp = getattr(cached, "block", cached).mlp
            first = Int8PrefillLinear(mlp.w1_t, mlp.b1, relu=True)
            second = Int8PrefillLinear(mlp.w2_t, mlp.b2)
            self.prefill_linears.append((first, second))
            metadata = {
                "block": index,
                "ffn1_weight_shape": list(mlp.w1_t.shape),
                "ffn2_weight_shape": list(mlp.w2_t.shape),
                "calls": 0,
                "arithmetic": "Approximate per-token W8A8 genuine multirow GEMM; FP32 dequant/bias",
            }

            def forward(x, first=first, second=second, metadata=metadata):
                batch, length, width = x.shape
                if batch * length <= 1:
                    raise RuntimeError("W8A8 FFN override is prompt-only, never GEMV")
                metadata["calls"] += 1
                metadata["input_shape"] = list(x.shape)
                value = first(x.reshape(batch * length, width).contiguous())
                return second(value).view(batch, length, -1)

            self.forwarders.append((mlp, forward))
            self.audit.append(metadata)


def build_prefill_ffn_proxy(original):
    return Int8FFNPrefillProxy(original)
