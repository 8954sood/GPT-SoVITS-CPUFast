"""Diagnostic attribution must preserve tensors and restore the selected engine."""

from types import SimpleNamespace

import pytest
import torch

from cpufast_profiling import ar_phase_ranges


class Transformer:
    def __init__(self):
        self.block = SimpleNamespace(
            qkv_w_t=torch.randn(8, 24),
            out_w_t=torch.randn(8, 8),
            mlp=SimpleNamespace(w1_t=torch.randn(8, 16), w2_t=torch.randn(16, 8)),
        )
        self.blocks = [SimpleNamespace(block=self.block)]

    def process_prompt(self, x):
        b = self.block
        q, k, v = torch.addmm(torch.zeros(24), x, b.qkv_w_t).chunk(3, dim=-1)
        a = torch.nn.functional.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None]
        )[0, 0]
        out = torch.addmm(torch.zeros(8), a, b.out_w_t)
        hidden = torch.relu(torch.addmm(torch.zeros(16), out, b.mlp.w1_t))
        return torch.addmm(torch.zeros(8), hidden, b.mlp.w2_t)

    def decode_next_token(self, x):
        return self.process_prompt(x)


def test_phase_shape_and_matrix_roles_preserve_results_and_restore_on_error():
    transformer = Transformer()
    model = SimpleNamespace(t2s_transformer=transformer)
    x = torch.randn(3, 8)
    expected = transformer.process_prompt(x)
    with ar_phase_ranges(model) as metadata:
        assert torch.equal(model.t2s_transformer.process_prompt(x), expected)
        assert torch.equal(model.t2s_transformer.decode_next_token(x), expected)
    assert model.t2s_transformer is transformer
    assert {r["phase"] for r in metadata["phases"]} == {"prefill", "decode"}
    for phase in ("prefill", "decode"):
        rows = [r for r in metadata["operators"] if r["phase"] == phase]
        assert {r["role"] for r in rows} >= {
            "qkv",
            "attention_out",
            "ffn1",
            "ffn_activation",
            "ffn2",
            "attention",
        }
        assert next(r for r in rows if r["role"] == "ffn1")["input_shapes"] == [
            [16],
            [3, 8],
            [8, 16],
        ]
    with pytest.raises(RuntimeError), ar_phase_ranges(model):
        raise RuntimeError("restore")
    assert model.t2s_transformer is transformer
