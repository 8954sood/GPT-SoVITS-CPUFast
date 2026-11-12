"""Standalone Linux ARM W8A8 GEMM correctness, not a TTS quality/speed test."""

import os
import unittest

import torch

from cpufast_prefill_int8_probe import Int8PrefillLinear, load_library


def reference(x, qw, scale, bias, relu=False):
    maximum = x.abs().amax(dim=1)
    sx = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
    qx = (x * sx.reciprocal()[:, None]).round().clamp(-127, 127).int()
    value = (qx @ qw.int().T).float() * sx[:, None] * scale[None, :] + bias
    return value.relu() if relu else value


class PrefillInt8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_library(os.environ["CPUFAST_INT8_PREFILL_LIBRARY"])
        torch.set_num_threads(2)

    def test_multirow_real_shapes_ragged_tails_and_quantized_reference(self):
        torch.manual_seed(4816)
        with torch.no_grad():
            for rows, inputs, outputs in (
                (256, 512, 2048),
                (256, 2048, 512),
                (256, 512, 1536),
                (256, 1536, 512),
                (7, 17, 13),
                (2, 1, 1),
            ):
                qw = torch.randint(-127, 128, (outputs, inputs), dtype=torch.int8)
                scale, bias = torch.rand(outputs) * 0.03 + 0.001, torch.randn(outputs)
                x = torch.randn(rows, inputs)
                x[0].zero_()
                before = x.clone(), qw.clone(), scale.clone(), bias.clone()
                packed = torch.ops.cpufast_cpp.prefill_int8_pack_rhs(qw)
                restored = packed.permute(0, 2, 1, 3).reshape(
                    ((outputs + 3) // 4) * 4, ((inputs + 15) // 16) * 16
                )
                self.assertTrue(torch.equal(restored[:outputs, :inputs], qw))
                self.assertEqual(int(restored[outputs:].count_nonzero()), 0)
                self.assertEqual(int(restored[:, inputs:].count_nonzero()), 0)
                for relu in (False, True):
                    expected = reference(x, qw, scale, bias, relu)
                    for threads in (1, 2):
                        torch.set_num_threads(threads)
                        actual = torch.ops.cpufast_cpp.prefill_int8_linear(
                            x, packed, scale, bias, inputs, relu
                        )
                        self.assertTrue(
                            torch.equal(actual, expected),
                            float((actual - expected).abs().max()),
                        )
                for value, saved in zip((x, qw, scale, bias), before):
                    self.assertTrue(torch.equal(value, saved))

    def test_source_weight_version_cache_and_bias_stay_authoritative(self):
        torch.manual_seed(42)
        with torch.no_grad():
            weight = torch.randn(17, 13).half().float().contiguous()
            bias = torch.randn(13)
            original = weight.clone()
            value = torch.randn(9, 17)
            linear = Int8PrefillLinear(weight, bias)
            actual = linear(value)
            self.assertTrue(
                torch.equal(
                    actual, reference(value, linear._quantized, linear._scale, bias)
                )
            )
            cached = linear._packed
            bias.add_(0.25)
            self.assertTrue(
                torch.equal(
                    linear(value),
                    reference(value, linear._quantized, linear._scale, bias),
                )
            )
            self.assertIs(linear._packed, cached)
            weight[0, 0] = 0.5
            linear(value)
            self.assertIsNot(linear._packed, cached)
            self.assertIs(linear.weight_t, weight)
            weight.copy_(original)
            self.assertTrue(
                torch.equal(
                    linear(value),
                    reference(value, linear._quantized, linear._scale, bias),
                )
            )
            self.assertTrue(torch.equal(weight, original))
            with self.assertRaises(ValueError):
                linear(value[:1])

    def test_zero_weights_and_nonexact_source_copy_rejection(self):
        with torch.no_grad():
            weight, bias = torch.zeros(17, 13), torch.randn(13)
            linear = Int8PrefillLinear(weight, bias)
            value = torch.randn(5, 17)
            actual = linear(value)
            self.assertTrue(torch.equal(actual, bias.expand_as(actual)))
            self.assertEqual(int(linear._quantized.count_nonzero()), 0)
            self.assertTrue(torch.equal(linear._scale, torch.full_like(bias, 1e-12)))
            bad = weight.clone()
            bad[0, 0] = 1.00001
            with self.assertRaises(ValueError):
                Int8PrefillLinear(bad, bias)(value)
            self.assertEqual(float(bad[0, 0]), float(torch.tensor(1.00001)))

    def test_invalid_geometry_nonfinite_and_scales_rejected(self):
        q = torch.ones(13, 17, dtype=torch.int8)
        packed = torch.ops.cpufast_cpp.prefill_int8_pack_rhs(q)
        scales, bias = torch.ones(13), torch.zeros(13)
        for x in (
            torch.ones(1, 17),
            torch.ones(2, 18),
            torch.full((2, 17), float("nan")),
            torch.full((2, 17), float("inf")),
        ):
            with self.assertRaises(RuntimeError):
                torch.ops.cpufast_cpp.prefill_int8_linear(
                    x, packed, scales, bias, 17, False
                )
        with self.assertRaises(RuntimeError):
            torch.ops.cpufast_cpp.prefill_int8_linear(
                torch.ones(2, 17), packed, scales.zero_(), bias, 17, False
            )


if __name__ == "__main__":
    unittest.main()
