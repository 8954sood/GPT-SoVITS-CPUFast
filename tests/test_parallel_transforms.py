"""Parallel channels preserve serial FP32 transform and epilogue values."""

import platform
import unittest

import torch
from cpufast_fft import FFTConv1d
from cpufast_winograd import WinogradConv1d


@unittest.skipUnless(
    platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only"
)
class ParallelTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cpufast_cpp_ops  # noqa: F401

        torch.set_num_threads(2)

    def test_winograd_input_output_and_strided_residual(self):
        torch.manual_seed(145)
        with torch.no_grad():
            for batch, channels, length, dilation in [
                (1, 12, 37, 1),
                (2, 24, 173, 3),
                (1, 48, 317, 5),
            ]:
                conv = torch.nn.Conv1d(
                    channels, channels, 3, padding=dilation, dilation=dilation
                ).eval()
                m = WinogradConv1d(conv, native=True)
                x = torch.randn(batch, channels, length)
                res = torch.randn(batch, channels, length + 4)[..., 2:-2]
                for active, residual in [
                    (False, None),
                    (True, None),
                    (False, res),
                    (True, res),
                ]:
                    m.parallel_transforms = False
                    expected = m(x, activate=active, residual=residual)
                    m.parallel_transforms = True
                    self.assertTrue(
                        torch.equal(expected, m(x, activate=active, residual=residual))
                    )

    def test_fft_finish_and_strided_residual(self):
        torch.manual_seed(147)
        with torch.no_grad():
            for batch, channels, length, dilation in [
                (1, 24, 73, 1),
                (2, 24, 279, 3),
                (1, 96, 773, 5),
            ]:
                conv = torch.nn.Conv1d(
                    channels, channels, 11, padding=5 * dilation, dilation=dilation
                ).eval()
                m = FFTConv1d(conv, backend="packed")
                m.output_layout = m.fused_finish = True
                x = torch.randn(batch, channels, length)
                res = torch.randn(batch, channels, length + 4)[..., 2:-2]
                for active, residual in [
                    (False, None),
                    (True, None),
                    (False, res),
                    (True, res),
                ]:
                    m.parallel_transforms = False
                    expected = m(x, activate=active, residual=residual)
                    m.parallel_transforms = True
                    self.assertTrue(
                        torch.equal(expected, m(x, activate=active, residual=residual))
                    )


if __name__ == "__main__":
    unittest.main()
