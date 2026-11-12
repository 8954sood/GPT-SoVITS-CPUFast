"""Only process_prompt may override FFN; restore methods even after errors."""

from types import SimpleNamespace

import pytest
import torch

import cpufast_prefill_probe


def test_prompt_only_override_restores_all_methods_on_success_and_error(monkeypatch):
    class Linear:
        def __init__(self, w, b, *, relu=False):
            self.w, self.b, self.relu = w, b, relu

        def __call__(self, x):
            y = torch.addmm(self.b, x, self.w)
            return y.relu() if self.relu else y

    monkeypatch.setattr(cpufast_prefill_probe, "PackedPrefillLinear", Linear)

    class MLP:
        def __init__(self):
            self.w1_t, self.w2_t = torch.randn(4, 8), torch.randn(8, 4)
            self.b1, self.b2 = torch.randn(8), torch.randn(4)

        def forward(self, x):
            y = torch.addmm(self.b1, x.reshape(-1, 4), self.w1_t).relu()
            return torch.addmm(self.b2, y, self.w2_t).view_as(x)

    class Transformer:
        def __init__(self):
            self.blocks = [SimpleNamespace(block=SimpleNamespace(mlp=MLP()))]
            self.fail = False

        def process_prompt(self, x):
            y = self.blocks[0].block.mlp.forward(x)
            if self.fail:
                raise RuntimeError("prompt fail")
            return y

        def decode_next_token(self, x):
            return self.blocks[0].block.mlp.forward(x)

    original = Transformer()
    mlp = original.blocks[0].block.mlp
    saved = mlp.forward
    proxy = cpufast_prefill_probe.build_prefill_ffn_proxy(original)
    x = torch.randn(1, 3, 4)
    with torch.no_grad():
        expected = original.process_prompt(x)
        assert torch.equal(proxy.process_prompt(x), expected)
        assert mlp.forward == saved and not proxy.busy
        assert proxy.audit[0]["calls"] == 1
        assert torch.equal(
            proxy.decode_next_token(x[:, :1]), original.decode_next_token(x[:, :1])
        )
        assert proxy.audit[0]["calls"] == 1
        original.fail = True
        with pytest.raises(RuntimeError, match="prompt fail"):
            proxy.process_prompt(x)
        assert mlp.forward == saved and not proxy.busy
