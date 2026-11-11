import torch, time, json, statistics
from pathlib import Path
from cpufast_cooktoom import CookToomConv1d, CookToomPhase
from cpufast_winograd import WinogradConv1d
from cpufast_optimizations import PhaseUpsample1d
import cpufast_cpp_ops

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
records = torch.load("/work/layer_captures.pt", weights_only=True, mmap=True)
rows = []
with torch.no_grad():
    for name in [
        "resblocks.0.convs1.0",
        "resblocks.3.convs1.0",
        "resblocks.3.convs1.2",
        "resblocks.6.convs1.0",
        "ups.0",
        "ups.1",
        "ups.2",
    ]:
        q = records[name]
        w = q["weight"]
        b = q["bias"]
        x = q["x"]
        phase = name.startswith("ups")
        if phase:
            raw = torch.nn.ConvTranspose1d(
                w.shape[0],
                w.shape[1],
                w.shape[-1],
                stride=q["stride"],
                padding=q["padding"],
            )
            methods = (
                ["phase", "cook4", "cook6"]
                if w.shape[-1] // q["stride"][0] < 4
                else ["phase", "cook4"]
            )
        else:
            raw = torch.nn.Conv1d(
                w.shape[1], w.shape[0], 3, padding=q["padding"], dilation=q["dilation"]
            )
            methods = ["winograd4", "cook6", "cook8"]
        raw.weight = torch.nn.Parameter(w)
        raw.bias = torch.nn.Parameter(b)
        raw.eval()
        ref = raw(x)
        for method in methods:
            if method == "phase":
                module = PhaseUpsample1d(raw)
            elif method == "winograd4":
                module = WinogradConv1d(raw, native=True)
            else:
                module = (
                    CookToomPhase(raw, m=int(method[-1]))
                    if phase
                    else CookToomConv1d(raw, m=int(method[-1]))
                )
            times = []
            cpu = []
            for i in range(5):
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
            Path("/repo/.experiments/probe_cook.json").write_text(
                json.dumps(rows, indent=2)
            )
            print(row, flush=True)
