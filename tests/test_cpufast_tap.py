import unittest

import torch
import torch.nn.functional as F

from cpufast_fft import FFTConv1d
from cpufast_tap import PackedSmallTap, TapConv1d


class TapConvTests(unittest.TestCase):
    def test_matches_original_for_dilations_and_fused_epilogue(self):
        torch.manual_seed(11)
        for channels in (4, 24, 48):
            for kernel, dilation in ((7, 1), (7, 3), (11, 5)):
                original = torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel,
                    padding=dilation * (kernel - 1) // 2,
                    dilation=dilation,
                ).eval()
                tap = TapConv1d(FFTConv1d(original).eval()).eval()
                x = torch.randn(1, channels, 71)
                residual = torch.randn_like(x)
                with torch.no_grad():
                    expected = F.leaky_relu(original(x), 0.1) + residual
                    actual = tap(x, activate=True, residual=residual)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-5)
                self.assertIs(tap.weight, original.weight)

    def test_packed_small_matches_and_invalidates(self):
        original = torch.nn.Conv1d(48, 48, 7, padding=3).eval()
        candidate = PackedSmallTap(FFTConv1d(original).eval()).eval()
        x, residual = torch.randn(1, 48, 73), torch.randn(1, 48, 73)
        with torch.no_grad():
            for _ in range(2):
                expected = F.leaky_relu(original(x), 0.1) + residual
                torch.testing.assert_close(
                    candidate(x, activate=True, residual=residual),
                    expected,
                    atol=3e-6,
                    rtol=1e-5,
                )
                original.weight.add_(0.01)
        self.assertIs(candidate.weight, original.weight)
        candidate.train()
        candidate(x.requires_grad_()).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_cache_invalidates_and_training_falls_back(self):
        original = torch.nn.Conv1d(4, 4, 7, padding=3).eval()
        tap = TapConv1d(FFTConv1d(original).eval()).eval()
        x = torch.randn(1, 4, 24)
        with torch.no_grad():
            tap(x)
            original.weight.add_(0.25)
            torch.testing.assert_close(tap(x), original(x), atol=3e-6, rtol=1e-5)
        tap.train()
        result = tap(x.requires_grad_())
        result.sum().backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
