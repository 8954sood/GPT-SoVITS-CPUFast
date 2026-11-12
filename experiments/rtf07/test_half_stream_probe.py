"""Compare the unlinked half-stream prototype bitwise with approved P operators.

Build using build_half_stream_probe.py and set CPUFAST_FAST_LIBRARY to that
separate library. These are correctness checks, never target-speed evidence.
"""

import platform
import unittest

import torch


@unittest.skipUnless(
    platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only"
)
class HalfStreamProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cpufast_fast_ops  # noqa: F401

    def test_bitwise_matches_materialized_fp16_products_and_fp32_unpack(self):
        torch.manual_seed(8192)
        # Frequency tails, row tails, multiple batches and actual channel sizes.
        for freq, batch, blocks, groups, k in (
            (1, 1, 1, 1, 21),
            (4, 1, 7, 2, 42),
            (5, 2, 9, 3, 84),
            (65, 1, 19, 32, 384),
            (129, 1, 92, 16, 192),
            (129, 2, 31, 8, 96),
        ):
            tiles = (batch * blocks + 7) // 8
            value = (torch.randn(3 * freq, tiles, k, 8) * 0.25).half()
            weight = (torch.randn(3 * freq, groups, k, 12) * 0.25).half()
            before = value.clone(), weight.clone()
            for threads in (1, 2):
                torch.set_num_threads(threads)
                products = torch.ops.cpufast_cpp.fft_hgemm(value, weight)
                expected = torch.ops.cpufast_cpp.fft_unpack_layout(
                    products, batch, blocks
                )
                actual = torch.ops.cpufast_cpp.fft_half_stream(
                    value, weight, batch, blocks
                )
                self.assertEqual(actual.dtype, torch.complex64)
                self.assertEqual(actual.shape, (batch, groups * 12, blocks, freq))
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(
                    torch.equal(actual, expected),
                    (freq, batch, blocks, groups, k, threads),
                )
            self.assertTrue(torch.equal(value, before[0]))
            self.assertTrue(torch.equal(weight, before[1]))

    def test_rejects_wrong_dtype_and_bad_output_geometry(self):
        value = torch.ones(15, 1, 8, 8).half()
        weight = torch.ones(15, 1, 8, 12).half()
        for arguments in (
            (value.float(), weight, 1, 1),
            (value, weight.float(), 1, 1),
            (value, weight, 1, 9),
            (value, weight, 0, 1),
        ):
            with self.assertRaises(RuntimeError):
                torch.ops.cpufast_cpp.fft_half_stream(*arguments)


if __name__ == "__main__":
    unittest.main()
