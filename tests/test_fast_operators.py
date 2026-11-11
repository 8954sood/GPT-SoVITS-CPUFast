"""Native numerical checks; run after building both ARM libraries."""

import unittest
import platform
from pathlib import Path

import torch

@unittest.skipUnless(platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only")
class FastOperatorsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (Path(__file__).resolve().parent.parent / "libcpufast_fast.so.json").exists():
            raise unittest.SkipTest("experimental native library has not been built")
        import cpufast_fast_ops  # noqa: F401

    def test_int8_linear_matches_quantized_reference_across_threads(self):
        torch.manual_seed(4)
        x = torch.randn(1, 1, 64)
        weight = torch.randn(32, 64)
        scale = weight.abs().amax(dim=1).clamp_min(1e-12) / 127
        q = (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        bias = torch.randn(32)
        sx = x.abs().max() / 127
        qx = (x / sx).round().clamp(-127, 127).to(torch.int32)
        expected = (qx @ q.to(torch.int32).T).float() * sx * scale + bias
        for threads in (1, 2):
            torch.set_num_threads(threads)
            actual = torch.ops.cpufast_cpp.int8_linear(x, q, scale, bias)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        with self.assertRaises(RuntimeError):
            torch.ops.cpufast_cpp.int8_linear(x, q, scale[:2], bias)

    def test_head_attention_matches_torch_with_per_head_mask(self):
        torch.manual_seed(5)
        q = torch.randn(1, 1, 32)
        keys = torch.randn(1, 4, 12, 8)
        values = torch.randn_like(keys)
        mask = torch.zeros(1, 4, 1, 7, dtype=torch.bool)
        mask[0, 2, 0, 0] = True
        mask[0, 3, 0, :] = True
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.view(1, 1, 4, 8).transpose(1, 2), keys[:, :, :7], values[:, :, :7], ~mask
        ).transpose(1, 2).reshape_as(q)
        for threads in (1, 2):
            torch.set_num_threads(threads)
            actual = torch.ops.cpufast_cpp.head_attn(q, keys, values, 7, 4, mask)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_half_fft_product_against_reference(self):
        torch.manual_seed(6)
        x = torch.randn(6, 2, 32, 8).half()
        w = torch.randn(6, 3, 32, 12).half()
        expected = torch.einsum("ftki,fnkj->ftnij", x.float(), w.float())
        actual = torch.ops.cpufast_cpp.fft_hgemm(x, w)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)


if __name__ == "__main__":
    unittest.main()
