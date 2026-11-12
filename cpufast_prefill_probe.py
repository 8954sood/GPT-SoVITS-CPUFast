"""Research-only disposable FP32 RHS cache and fused-bias/ReLU prefill GEMM.

Never enabled by the engine default. Single-row/decode inputs are rejected.
"""

import hashlib
import json
from pathlib import Path

import torch


def load_library(path):
    root = Path(__file__).resolve().parent
    path = Path(path)
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    digest = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
    if (
        manifest["torch_version"] != torch.__version__
        or digest(path) != manifest["library_sha256"]
        or any(
            digest(root / name) != expected
            for name, expected in manifest["sources"].items()
        )
    ):
        raise RuntimeError("Prefill research library/source mismatch")
    torch.ops.load_library(str(path))


class PackedPrefillLinear:
    def __init__(self, weight_t, bias, *, relu=False):
        self.weight_t, self.bias, self.relu = weight_t, bias, relu
        self._key = self._packed = None

    def __call__(self, x):
        if (
            x.ndim != 2
            or x.shape[0] <= 1
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cpu")
        ):
            raise ValueError(
                "Research packed FP32 prefill requires multi-row inference; no decode/GEMV route"
            )
        key = (
            id(self.weight_t),
            self.weight_t._version,
            self.weight_t.data_ptr(),
            self.weight_t.dtype,
            self.weight_t.device,
        )
        if key != self._key:
            self._packed = torch.ops.cpufast_cpp.prefill_pack_rhs(self.weight_t)
            self._key = key
        return torch.ops.cpufast_cpp.prefill_packed_linear(
            x, self._packed, self.bias, self.relu
        )


class PackedFFNPrefillProxy:
    """Opt-in FFN replacement during process_prompt only; decode is delegated.

    The selected L transformer is a Python ScriptClass instance. Its original
    block/attention/QKV/out/norm/cache methods remain authoritative. Reentrant
    prompt calls are rejected because temporary Python method overrides are
    scoped to one request. Parameters and state_dict never change.
    """

    def __init__(self, original):
        self.original = original
        self.busy = False
        self.prefill_calls = 0
        self.forwarders = []
        self.prefill_linears = []
        self.audit = []
        for index, cached in enumerate(original.blocks):
            block = getattr(cached, "block", cached)
            mlp = block.mlp
            ff1 = PackedPrefillLinear(mlp.w1_t, mlp.b1, relu=True)
            ff2 = PackedPrefillLinear(mlp.w2_t, mlp.b2)
            self.prefill_linears.append((ff1, ff2))
            row = {
                "block": index,
                "ffn1_weight_shape": list(mlp.w1_t.shape),
                "ffn2_weight_shape": list(mlp.w2_t.shape),
                "calls": 0,
            }

            def forward(x, first=ff1, second=ff2, metadata=row):
                batch, length, width = x.shape
                if batch * length <= 1:
                    raise RuntimeError("Packed FFN override is prompt-only, not GEMV")
                metadata["calls"] += 1
                metadata["input_shape"] = list(x.shape)
                hidden = first(x.reshape(batch * length, width).contiguous())
                return second(hidden).view(batch, length, -1)

            self.forwarders.append((mlp, forward))
            self.audit.append(row)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def process_prompt(self, *args, **kwargs):
        if self.busy:
            raise RuntimeError("Reentrant packed-prefill experiment is unsupported")
        if args[0].shape[0] * args[0].shape[1] <= 1:
            raise ValueError("Packed prefill requires multiple prompt rows")
        saved = []
        before = [row["calls"] for row in self.audit]
        self.busy = True
        try:
            for mlp, forward in self.forwarders:
                saved.append((mlp, mlp.forward))
                mlp.forward = forward
            result = self.original.process_prompt(*args, **kwargs)
            if any(row["calls"] != count + 1 for row, count in zip(self.audit, before)):
                raise RuntimeError(
                    "Packed FFN override did not execute once per actual prompt block"
                )
            self.prefill_calls += 1
            return result
        finally:
            for mlp, forward in reversed(saved):
                mlp.forward = forward
            self.busy = False

    def decode_next_token(self, *args, **kwargs):
        if self.busy:
            raise RuntimeError("Concurrent decode/prefill experiment is unsupported")
        return self.original.decode_next_token(*args, **kwargs)


def build_prefill_ffn_proxy(original):
    return PackedFFNPrefillProxy(original)
