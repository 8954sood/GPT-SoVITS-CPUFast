"""Check head-major native attention including masks and all-masked heads."""

import platform
import unittest
import torch
import torch.nn.functional as F


@unittest.skipUnless(
    platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only"
)
class HeadAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cpufast_cpp_ops  # noqa: F401

    def test_matches_pytorch_for_masks_and_threads(self):
        torch.manual_seed(5)
        q = torch.randn(1, 1, 32)
        keys = torch.randn(1, 4, 12, 8)
        values = torch.randn_like(keys)
        for mask in (
            None,
            torch.zeros(1, 1, 1, 7, dtype=torch.bool),
            torch.zeros(1, 4, 1, 7, dtype=torch.bool),
        ):
            if mask is not None:
                mask[..., 0] = True
                if mask.numel() == 28:
                    mask[:, 3, :, :] = True
            expected = (
                F.scaled_dot_product_attention(
                    q.view(1, 1, 4, 8).transpose(1, 2),
                    keys[:, :, :7],
                    values[:, :, :7],
                    ~mask if mask is not None else None,
                )
                .transpose(1, 2)
                .reshape_as(q)
            )
            for threads in (1, 2):
                torch.set_num_threads(threads)
                actual = torch.ops.cpufast_cpp.head_attn_fp32(
                    q, keys, values, 7, 4, mask
                )
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
