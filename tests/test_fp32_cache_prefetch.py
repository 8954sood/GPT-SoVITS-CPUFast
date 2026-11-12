"""Explicit FP32-cache prefetch preserves non-half-representable coefficients."""

import platform
import unittest

import torch


@unittest.skipUnless(
    platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only"
)
class FP32PrefetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cpufast_cpp_ops  # noqa: F401

    def test_exact_against_fp32_no_prefetch(self):
        torch.manual_seed(207)
        for outputs, inputs in [
            (512, 512),
            (1536, 512),
            (2048, 512),
            (512, 2048),
            (1040, 512),
        ]:
            original = torch.randn(outputs, inputs)
            self.assertFalse(torch.equal(original, original.half().float()))
            packed = (
                original.reshape(outputs // 8, 8, inputs).transpose(1, 2).contiguous()
            )
            x = torch.randn(1, 1, inputs)
            bias = torch.randn(outputs)
            before = packed.clone()
            for threads in (1, 2):
                torch.set_num_threads(threads)
                expected = torch.ops.cpufast_cpp.lossless_linear_f32(x, packed, bias)
                for distance in (0, 64, 128, 256):
                    actual = torch.ops.cpufast_cpp.lossless_linear_f32_prefetch(
                        x, packed, bias, distance
                    )
                    self.assertTrue(
                        torch.equal(actual, expected),
                        (outputs, inputs, threads, distance),
                    )
            self.assertTrue(torch.equal(packed, before))


if __name__ == "__main__":
    unittest.main()
