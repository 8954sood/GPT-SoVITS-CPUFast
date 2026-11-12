"""Diagnostic module labels, installed exclusively for excluded diagnostic inference."""

from contextlib import ExitStack, contextmanager

import torch
from torch.utils._python_dispatch import TorchDispatchMode


@contextmanager
def module_range(module, label):
    original = module.forward
    overridden = "forward" in module.__dict__
    shapes = {}
    metadata = {
        "scope": "Extra profiled synthesis only; no instrumentation in timed samples",
        "range": label,
        "weight_shape": list(module.weight.shape),
        "weight_dtype": str(module.weight.dtype),
        "negative_eligibility_cache": getattr(module, "cache_ineligible", False),
    }

    def forward(x, *args, **kwargs):
        if "torch_num_threads_observed" not in metadata:
            metadata["torch_num_threads_observed"] = torch.get_num_threads()
        shape = tuple(x.shape)
        shapes[shape] = shapes.get(shape, 0) + 1
        with torch.profiler.record_function(metadata["range"]):
            return original(x, *args, **kwargs)

    module.forward = forward
    try:
        yield metadata
    finally:
        if overridden:
            module.forward = original
        else:
            del module.forward
        metadata["input_shapes"] = [
            {"shape": list(shape), "calls": count} for shape, count in shapes.items()
        ]


@contextmanager
def predictor_range(module):
    with module_range(module, "cpufast_predictor_projection") as metadata:
        yield metadata
    metadata["packed_half_classifier"] = getattr(module, "_packed", None) is not None


@contextmanager
def decoder_tap_ranges(decoder):
    from cpufast_tap import TapConv1d

    metadata = []
    with ExitStack() as stack:
        for name, module in decoder.named_modules():
            if isinstance(module, TapConv1d):
                channels, _, kernel = module.weight.shape
                label = f"cpufast_decoder_tap_C{channels}_K{kernel}_D{module.dilation}"
                row = stack.enter_context(module_range(module, label))
                row.update(
                    {
                        "module": name,
                        "dilation": module.dilation,
                        "padding": module.padding,
                    }
                )
                metadata.append(row)
        yield metadata


@contextmanager
def decoder_fft_ranges(decoder):
    from cpufast_fft import FFTConv1d

    metadata = []
    with ExitStack() as stack:
        for name, module in decoder.named_modules():
            if isinstance(module, FFTConv1d):
                cout, cin, kernel = module.weight.shape
                label = f"cpufast_decoder_fft_Cin{cin}_Cout{cout}_K{kernel}_D{module.dilation}_N{module.n_fft}"
                row = stack.enter_context(module_range(module, label))
                row.update(
                    module=name,
                    dilation=module.dilation,
                    n_fft=module.n_fft,
                    stream_products=module.stream_products,
                    fast_half=module.fast_half,
                    half_stream_products=module.half_stream_products,
                )
                metadata.append(row)
        yield metadata


@contextmanager
def ar_phase_ranges(model):
    """Attribute real AR operators without retaining global profiler input tensors.

    Installed only for the separate excluded profile. Scripted prefill remains
    the original FP32 block; decode remains the original selected runtime.
    Matrix storage identities distinguish QKV/out/FFN from acoustic addmm.
    """
    original = model.t2s_transformer
    roles = {}
    rows = {}
    phases = {}
    for cached in original.blocks:
        block = getattr(cached, "block", cached)
        for role, matrix in (
            ("qkv", block.qkv_w_t),
            ("attention_out", block.out_w_t),
            ("ffn1", block.mlp.w1_t),
            ("ffn2", block.mlp.w2_t),
        ):
            roles[matrix.data_ptr()] = role
        for name, role in (
            ("qkv_q", "qkv"),
            ("out_q", "attention_out"),
            ("ff1_q", "ffn1"),
            ("ff2_q", "ffn2"),
        ):
            matrix = getattr(cached, name, None)
            if matrix is not None:
                roles[matrix.data_ptr()] = role
    for first, second in getattr(original, "prefill_linears", []):
        for linear, role in ((first, "ffn1"), (second, "ffn2")):
            if linear._packed is not None:
                roles[linear._packed.data_ptr()] = role

    class Observe(TorchDispatchMode):
        def __init__(self, phase):
            super().__init__()
            self.phase = phase

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            role = None
            if name in (
                "aten.addmm.default",
                "aten.mm.default",
                "cpufast_cpp.int8_linear.default",
                "cpufast_cpp.prefill_packed_linear.default",
            ):
                matrix_index = 2 if name == "aten.addmm.default" else 1
                role = roles.get(args[matrix_index].data_ptr(), "unattributed_matrix")
            elif "attention" in name or name == "cpufast_cpp.head_attn.default":
                role = "attention"
            elif name == "aten.relu.default":
                role = "ffn_activation"
            if role is None:
                return func(*args, **(kwargs or {}))
            shapes = tuple(
                tuple(value.shape) for value in args if isinstance(value, torch.Tensor)
            )
            key = self.phase, role, name, shapes
            rows[key] = rows.get(key, 0) + 1
            with torch.profiler.record_function(f"cpufast_ar_{self.phase}_{role}"):
                return func(*args, **(kwargs or {}))

    class Proxy:
        def __getattr__(self, name):
            return getattr(original, name)

        def call(self, phase, method, args, kwargs):
            shape = tuple(args[0].shape)
            key = phase, shape
            phases[key] = phases.get(key, 0) + 1
            with torch.profiler.record_function(f"cpufast_ar_{phase}"), Observe(phase):
                return method(*args, **kwargs)

        def process_prompt(self, *args, **kwargs):
            return self.call("prefill", original.process_prompt, args, kwargs)

        def decode_next_token(self, *args, **kwargs):
            return self.call("decode", original.decode_next_token, args, kwargs)

    metadata = {
        "scope": "Excluded diagnostic only; dispatch labels add overhead, do not use profile wall time for RTF",
        "classification": "Actual runtime process_prompt/decode scopes and matrix-storage identity; acoustic addmm excluded",
    }
    model.t2s_transformer = Proxy()
    try:
        yield metadata
    finally:
        model.t2s_transformer = original
        metadata["phases"] = [
            {"phase": phase, "input_shape": list(shape), "calls": count}
            for (phase, shape), count in phases.items()
        ]
        metadata["operators"] = [
            {
                "phase": phase,
                "role": role,
                "operator": name,
                "input_shapes": [list(shape) for shape in shapes],
                "calls": count,
            }
            for (phase, role, name, shapes), count in rows.items()
        ]
