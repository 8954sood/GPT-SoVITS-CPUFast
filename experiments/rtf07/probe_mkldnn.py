import torch, time, json
from pathlib import Path

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
rows = []
weights = torch.load(
    "/baseline/models/s1v3.ckpt", map_location="cpu", weights_only=False, mmap=True
)["weight"]
selected = []
seen = set()
for name, w in weights.items():
    if (
        w.ndim == 2
        and w.shape[0] % 16 == 0
        and w.shape[1] % 4 == 0
        and ("linear" in name or "in_proj" in name or "out_proj" in name)
        and tuple(w.shape) not in seen
    ):
        selected.append((name, w.float()))
        seen.add(tuple(w.shape))
with torch.no_grad():
    for name, w in selected:
        for m in [128, 320]:
            x = torch.randn(m, w.shape[1])
            bias = torch.zeros(w.shape[0])
            wt = w.t().contiguous()
            ref = torch.addmm(bias, x, wt)
            wm = w.to_mkldnn()
            bm = bias.to_mkldnn()
            for label in ["torch", "mkldnn"]:
                fn = (
                    (lambda: torch.addmm(bias, x, wt))
                    if label == "torch"
                    else (
                        lambda: torch._C._nn.mkldnn_linear(
                            x.to_mkldnn(), wm, bm
                        ).to_dense()
                    )
                )
                times = []
                cpu = []
                try:
                    for i in range(6):
                        t = time.perf_counter()
                        c = time.process_time()
                        out = fn()
                        cpu.append(time.process_time() - c)
                        times.append(time.perf_counter() - t)
                    delta = (out - ref).double()
                    row = {
                        "shape": [m, *w.shape],
                        "method": label,
                        "seconds": times,
                        "cpu": cpu,
                        "max_abs": float(delta.abs().max()),
                        "snr_db": None
                        if float(delta.square().mean()) == 0
                        else float(
                            10
                            * torch.log10(
                                ref.double().square().mean() / delta.square().mean()
                            )
                        ),
                    }
                except Exception as e:
                    row = {"shape": [m, *w.shape], "method": label, "error": str(e)}
                rows.append(row)
                Path("/repo/.experiments/probe_mkldnn.json").write_text(
                    json.dumps(rows, indent=2)
                )
                print(row, flush=True)
