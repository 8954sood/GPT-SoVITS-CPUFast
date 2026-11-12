"""Validate the proposed 672-wide decoder's 42/21 channel native tails."""

import platform
import unittest

import torch
import torch.nn.functional as F

from cpufast_winograd import WinogradConv1d


@unittest.skipUnless(
    platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only"
)
class WinogradOddChannelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cpufast_cpp_ops  # noqa: F401

    def test_channel_tails_match_convolution_and_parallel_is_bit_exact(self):
        torch.manual_seed(672)
        with torch.no_grad():
            for channels in (42, 21):
                for batch, dilation in ((1, 1), (2, 3), (1, 5)):
                    layer = torch.nn.Conv1d(
                        channels, channels, 3, padding=dilation, dilation=dilation
                    ).eval()
                    original = layer.weight.clone(), layer.bias.clone()
                    module = WinogradConv1d(layer, native=True)
                    value = torch.randn(batch, channels, 137)
                    residual = torch.randn(batch, channels, 141)[..., 2:-2]
                    reference = F.leaky_relu(layer(value), 0.1) + residual
                    for threads in (1, 2):
                        torch.set_num_threads(threads)
                        module.parallel_transforms = False
                        serial = module(value, activate=True, residual=residual)
                        module.parallel_transforms = True
                        parallel = module(value, activate=True, residual=residual)
                        self.assertEqual(parallel.shape, (batch, channels, 137))
                        self.assertTrue(torch.equal(parallel, serial))
                        torch.testing.assert_close(
                            parallel, reference, rtol=2e-4, atol=2e-4
                        )
                    self.assertTrue(torch.equal(layer.weight, original[0]))
                    self.assertTrue(torch.equal(layer.bias, original[1]))


if __name__ == "__main__":
    unittest.main()
