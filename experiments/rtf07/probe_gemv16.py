import torch, time, json
from pathlib import Path
import cpufast_cpp_ops

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
rows = []
for n, k in [(1536, 512), (512, 512), (2048, 512), (512, 2048)]:
    torch.manual_seed(42)
    w = torch.randn(n, k).half()
    x = torch.randn(1, 1, k)
    bias = torch.randn(n)
    reference = None
    for block in [8, 16]:
        packed = w.reshape(n // block, block, k).transpose(1, 2).contiguous()
        fn = (
            torch.ops.cpufast_cpp.lossless_linear
            if block == 8
            else torch.ops.cpufast_cpp.lossless_linear16
        )
        times = []
        cpu = []
        for j in range(6):
            t = time.perf_counter()
            c = time.process_time()
            for i in range(100):
                out = fn(x, packed, bias)
            times.append((time.perf_counter() - t) / 100)
            cpu.append((time.process_time() - c) / 100)
        if reference is None:
            reference = out
        row = {
            "n": n,
            "k": k,
            "block": block,
            "seconds": times,
            "cpu": cpu,
            "exact": torch.equal(out, reference),
            "max_abs": float((out - reference).abs().max()),
        }
        rows.append(row)
        print(row, flush=True)
Path("/repo/.experiments/probe_gemv16.json").write_text(json.dumps(rows, indent=2))
