"""Single-worker FP32 ARM operator reading an exactly representable FP16 cache."""

import os

import torch

from native_cpu import load_library

_LIBRARY = None
WORKERS = int(os.environ.get("CPUFAST_AR_WORKERS", "1"))
if WORKERS not in (1, 2):
    raise ValueError("Expected one or two native AR workers")


def _linear(x, weight, bias):
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = load_library()
    if (
        x.device.type != "cpu"
        or weight.device.type != "cpu"
        or bias.device.type != "cpu"
        or x.dtype != torch.float32
        or weight.dtype != torch.float16
        or bias.dtype != torch.float32
        or weight.ndim != 3
        or weight.shape[-1] != 8
        or weight.shape[1] % 4
        or bias.numel() != weight.shape[0] * 8
        or x.numel() != weight.shape[1]
        or x.shape[-1] != weight.shape[1]
        or not (x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous())
    ):
        raise ValueError("Expected one FP32 token and an exact ARM coefficient cache")
    output = x.new_empty((*x.shape[:-1], weight.shape[0] * 8))
    _LIBRARY.gsv_half_gemv(
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        output.data_ptr(),
        output.numel(),
        weight.shape[1],
        WORKERS,
    )
    return output


_DEFINITION = torch.library.Library("cpufast_native", "DEF")
_DEFINITION.define("lossless_linear(Tensor x, Tensor weight, Tensor bias) -> Tensor")
_IMPLEMENTATION = torch.library.Library("cpufast_native", "IMPL", "CPU")
_IMPLEMENTATION.impl("lossless_linear", _linear)
