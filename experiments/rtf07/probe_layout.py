import torch, time, json, statistics
from pathlib import Path
from cpufast_cooktoom import CookToomConv1d
from cpufast_fft import FFTConv1d
import cpufast_cpp_ops

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
records = torch.load("/work/layer_captures.pt", weights_only=True, mmap=True)
rows = []
with torch.no_grad():
    for name in [
        "resblocks.7.convs1.0",
        "resblocks.7.convs1.2",
        "resblocks.8.convs1.0",
        "resblocks.8.convs1.2",
        "resblocks.10.convs1.0",
        "resblocks.10.convs1.2",
        "resblocks.11.convs1.0",
        "resblocks.11.convs1.2",
        "resblocks.13.convs1.0",
        "resblocks.13.convs1.2",
        "resblocks.14.convs1.0",
        "resblocks.14.convs1.2",
    ]:
        q = records[name]
        w = q["weight"]
        b = q["bias"]
        x = q["x"]
        raw = torch.nn.Conv1d(
            w.shape[1],
            w.shape[0],
            w.shape[-1],
            padding=q["padding"],
            dilation=q["dilation"],
        )
        raw.weight = torch.nn.Parameter(w)
        raw.bias = torch.nn.Parameter(b)
        raw.eval()
        ref = raw(x)
        for method in ["packed", "packed_layout"]:
            if method in ("packed", "packed_layout"):
                module = FFTConv1d(raw, backend="packed")
                module.output_layout = method == "packed_layout"
            else:
                module = CookToomConv1d(raw, m=int(method[-1]), native=True)
            times = []
            cpu = []
            for i in range(4):
                t = time.perf_counter()
                c = time.process_time()
                out = module(x)
                times.append(time.perf_counter() - t)
                cpu.append(time.process_time() - c)
            delta = (out - ref).double()
            row = {
                "name": name,
                "method": method,
                "seconds": times,
                "cpu": cpu,
                "max_abs": float(delta.abs().max()),
                "snr_db": float(
                    10
                    * torch.log10(ref.double().square().mean() / delta.square().mean())
                ),
            }
            rows.append(row)
            Path("/repo/.experiments/probe_packed_layout.json").write_text(
                json.dumps(rows, indent=2)
            )
            print(row, flush=True)
