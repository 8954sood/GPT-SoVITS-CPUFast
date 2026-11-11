import torch, time, json, statistics
import cpufast_cpp_ops

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
rows = []
for length in [128, 256, 512]:
    for masked in [False, True]:
        torch.manual_seed(42)
        q = torch.randn(1, 1, 512)
        keys = torch.randn(1, 1024, 512)
        values = torch.randn_like(keys)
        mask = (torch.arange(length) % 7 == 0).view(1, 1, 1, -1) if masked else None

        def ref():
            return (
                torch.nn.functional.scaled_dot_product_attention(
                    q.view(1, 1, 16, 32).transpose(1, 2),
                    keys[:, :length].view(1, length, 16, 32).transpose(1, 2),
                    values[:, :length].view(1, length, 16, 32).transpose(1, 2),
                    ~mask if mask is not None else None,
                )
                .transpose(1, 2)
                .reshape(1, 1, 512)
            )

        def native():
            return torch.ops.cpufast_cpp.token_attention(
                q, keys, values, length, 16, mask
            )

        expected = ref()
        for label, fn in [("torch", ref), ("native", native)]:
            times = []
            cpu = []
            for r in range(5):
                start = time.perf_counter()
                c = time.process_time()
                for i in range(100):
                    out = fn()
                times.append((time.perf_counter() - start) / 100)
                cpu.append((time.process_time() - c) / 100)
            delta = (out - expected).double()
            row = {
                "length": length,
                "masked": masked,
                "method": label,
                "seconds": times,
                "cpu": cpu,
                "max_abs": float(delta.abs().max()),
                "snr_db": None
                if not delta.square().mean()
                else float(
                    10
                    * torch.log10(
                        expected.double().square().mean() / delta.square().mean()
                    )
                ),
            }
            rows.append(row)
            print(row, flush=True)
from pathlib import Path

Path("/repo/.experiments/probe_attention.json").write_text(json.dumps(rows, indent=2))
