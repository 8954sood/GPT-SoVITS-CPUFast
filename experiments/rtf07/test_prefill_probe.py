"""Correctness only for the isolated FP32 packed-prefill implementation."""

import os
import platform
import unittest

import torch

from cpufast_prefill_probe import PackedPrefillLinear, load_library


@unittest.skipUnless(
    platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only"
)
class PrefillProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_library(os.environ["CPUFAST_PREFILL_LIBRARY"])

    def test_observed_prefill_and_ragged_shapes_bias_activation_and_source_immutability(
        self,
    ):
        torch.manual_seed(256512)
        with torch.no_grad():
            for rows, k, n, relu in (
                (256, 512, 1536, False),
                (256, 512, 512, False),
                (256, 512, 2048, True),
                (256, 2048, 512, False),
                (3, 12, 13, True),
            ):
                weight = torch.randn(k, n) * 0.03
                bias = torch.randn(n) * 0.01
                value = torch.randn(rows, k)
                before = weight.clone(), bias.clone(), value.clone()
                cached = PackedPrefillLinear(weight, bias, relu=relu)
                expected = torch.addmm(bias, value, weight)
                if relu:
                    expected = torch.relu(expected)
                for threads in (1, 2):
                    torch.set_num_threads(threads)
                    actual = cached(value)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
                packed = cached._packed
                self.assertIs(cached(value).dtype, torch.float32)
                self.assertIs(cached._packed, packed)
                self.assertTrue(
                    all(
                        torch.equal(p, b) for p, b in zip((weight, bias, value), before)
                    )
                )
                key = cached._key
                weight[0, 0].add_(0.1)
                updated = cached(value)
                self.assertNotEqual(cached._key, key)
                reference = torch.addmm(bias, value, weight)
                if relu:
                    reference = torch.relu(reference)
                torch.testing.assert_close(updated, reference, rtol=1e-4, atol=1e-4)

    def test_single_row_decode_route_and_wrong_dtype_are_rejected(self):
        cached = PackedPrefillLinear(torch.ones(8, 13), torch.zeros(13))
        with torch.no_grad():
            with self.assertRaisesRegex(ValueError, "no decode/GEMV"):
                cached(torch.ones(1, 8))
            with self.assertRaises(RuntimeError):
                cached(torch.ones(3, 8, dtype=torch.float16))


if __name__ == "__main__":
    unittest.main()
