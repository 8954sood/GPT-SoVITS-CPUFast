"""Streaming FFT products preserve the former FP32 GEMM/combination order."""
import platform
import unittest

import torch
from cpufast_fft import FFTConv1d


@unittest.skipUnless(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Linux ARM64 only')
class FFTStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cpufast_cpp_ops  # noqa: F401

    def test_packed_products_are_bit_exact(self):
        torch.manual_seed(52)
        for batch, blocks, channels, outputs, freq in [(1, 1, 12, 12, 33), (1, 13, 24, 36, 65), (2, 11, 48, 24, 129)]:
            tiles = (batch * blocks + 7) // 8
            x = torch.randn(3 * freq, tiles, channels, 8)
            w = torch.randn(3 * freq, outputs // 12, channels, 12)
            for threads in (1, 2):
                torch.set_num_threads(threads)
                expected = torch.ops.cpufast_cpp.fft_unpack_layout(torch.ops.cpufast_cpp.fft_packed_gemm(x, w), batch, blocks)
                actual = torch.ops.cpufast_cpp.fft_packed_stream(x, w, batch, blocks)
                self.assertTrue(torch.equal(actual, expected), (batch, blocks, channels, freq, threads))

    def test_full_convolution_and_epilogues_are_bit_exact(self):
        torch.manual_seed(73)
        with torch.no_grad():
            for batch, length, channels, kernel, dilation in [(1, 137, 24, 7, 1), (2, 279, 48, 11, 5), (1, 773, 96, 11, 3)]:
                conv = torch.nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) * dilation // 2, dilation=dilation).eval()
                m = FFTConv1d(conv, backend='packed')
                m.output_layout = m.fused_finish = True
                x = torch.randn(batch, channels, length)
                residual = torch.randn(batch, channels, length + 4)[..., 2:-2]
                for threads in (1, 2):
                    torch.set_num_threads(threads)
                    for active, res in [(False, None), (True, None), (False, residual), (True, residual)]:
                        m.stream_products = False
                        expected = m(x, activate=active, residual=res)
                        m.stream_products = True
                        actual = m(x, activate=active, residual=res)
                        self.assertTrue(torch.equal(actual, expected), (batch, length, channels, active, res is not None))


if __name__ == '__main__':
    unittest.main()
