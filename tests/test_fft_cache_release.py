"""Cache lifecycle and arithmetic equivalence; not a performance experiment."""

import platform
import unittest

import torch

from cpufast_fft import FFTConv1d
from cpufast_fft_cache_release import (
    install_half_cache_release,
    live_spectral_cache_bytes,
)


@unittest.skipUnless(
    platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only"
)
class HalfCacheReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cpufast_fast_ops  # noqa: F401

    def test_half_fp32_toggles_and_parameter_mutation_are_bitwise_equivalent(self):
        torch.manual_seed(2150)
        with torch.no_grad():
            for stream in (False, True):
                conv = torch.nn.Conv1d(96, 96, 11, padding=15, dilation=3).eval()
                original = [p.clone() for p in conv.parameters()]
                reference = FFTConv1d(conv, backend="packed")
                candidate = FFTConv1d(conv, backend="packed")
                for layer in (reference, candidate):
                    layer.output_layout = layer.fused_finish = (
                        layer.parallel_transforms
                    ) = True
                    layer.fast_half = True
                    layer.half_stream_products = stream
                install_half_cache_release(candidate)
                value = torch.randn(2, 96, 137)
                residual = torch.randn_like(value)
                for threads in (1, 2):
                    torch.set_num_threads(threads)

                    def check(
                        reference=reference,
                        candidate=candidate,
                        value=value,
                        residual=residual,
                    ):
                        expected = reference(value, activate=True, residual=residual)
                        actual = candidate(value, activate=True, residual=residual)
                        self.assertTrue(torch.equal(actual, expected))

                    check()
                    self.assertIsNone(candidate._packed)
                    self.assertIsNotNone(candidate._packed_half)
                    self.assertEqual(
                        live_spectral_cache_bytes(candidate)["fp32_derived_bytes"], 0
                    )
                    conv.weight[0, 0, 0].add_(0.01)
                    check()
                    candidate.half_cache_release_enabled = False
                    check()
                    self.assertIsNotNone(candidate._packed)
                    candidate.half_cache_release_enabled = True
                    check()
                    self.assertIsNone(candidate._packed)
                    for layer in (reference, candidate):
                        layer.fast_half = False
                        layer.half_stream_products = False
                        layer.stream_products = True
                    check()
                    self.assertIsNotNone(candidate._packed)
                    conv.bias[0].add_(0.01)
                    check()
                    for layer in (reference, candidate):
                        layer.fast_half = True
                        layer.stream_products = False
                        layer.half_stream_products = stream
                    check()
                    self.assertIsNone(candidate._packed)
                    conv.weight.copy_(original[0])
                    conv.bias.copy_(original[1])
                    check()
                self.assertIs(candidate.weight, conv.weight)
                self.assertIs(candidate.bias, conv.bias)
                self.assertTrue(
                    all(torch.equal(p, b) for p, b in zip(conv.parameters(), original))
                )

    def test_invalid_install_is_rejected_without_mutation(self):
        layer = FFTConv1d(torch.nn.Conv1d(96, 96, 7).eval(), backend="packed")
        with self.assertRaises(ValueError):
            install_half_cache_release(layer)
        self.assertNotIn("half_cache_release_enabled", layer.__dict__)


if __name__ == "__main__":
    unittest.main()
