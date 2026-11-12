"""Isolate attention execution without changing INT8 projections or KV layout."""

import platform
import unittest

import torch


@unittest.skipUnless(
    platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only"
)
class Int8SDPATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cpufast_fast_ops  # noqa: F401
        from AR.models.t2s_model import T2SBlock, T2SMLP
        from cpufast_fast_ar import CppLosslessBlock

        torch.manual_seed(142)

        def weight(rows, cols):
            return (torch.randn(rows, cols) * 0.03).half().float()

        mlp = T2SMLP(
            weight(128, 32), torch.zeros(128), weight(32, 128), torch.zeros(32)
        )
        original = T2SBlock(
            4,
            32,
            mlp,
            weight(96, 32),
            torch.zeros(96),
            weight(32, 32),
            torch.zeros(32),
            torch.ones(32),
            torch.zeros(32),
            1e-5,
            torch.ones(32),
            torch.zeros(32),
            1e-5,
        )
        cls.legacy = CppLosslessBlock(original, False, True)
        cls.sdpa = CppLosslessBlock(original, False, True, True)

    def test_default_native_and_optin_sdpa_have_same_cache_and_close_outputs(self):
        before = self.sdpa.block.qkv_w_t.clone()
        for threads in (1, 2):
            torch.set_num_threads(threads)
            for mask in (None, torch.zeros(1, 4, 1, 257, dtype=torch.bool)):
                if mask is not None:
                    mask[..., 0] = True
                    mask[:, 3] = True
                value = torch.randn(1, 1, 32)
                keys, values = torch.randn(1, 4, 300, 8), torch.randn(1, 4, 300, 8)
                k0, v0, k1, v1 = (
                    keys.clone(),
                    values.clone(),
                    keys.clone(),
                    values.clone(),
                )
                old, old_k, old_v = self.legacy.decode_next_token(
                    value, k0, v0, 256, mask
                )
                new, new_k, new_v = self.sdpa.decode_next_token(
                    value, k1, v1, 256, mask
                )
                torch.testing.assert_close(new, old, rtol=1e-3, atol=1e-4)
                self.assertTrue(torch.equal(old_k, new_k))
                self.assertTrue(torch.equal(old_v, new_v))
                self.assertTrue(torch.equal(new_k[:, :, 257:], keys[:, :, 257:]))
        self.assertTrue(torch.equal(before, self.sdpa.block.qkv_w_t))


if __name__ == "__main__":
    unittest.main()
