import torch, time, json, statistics
from pathlib import Path

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Strassen:
    def __init__(self, a, depth):
        self.a = a
        self.depth = depth
        if depth:
            m, k = a.shape[-2:]
            a11 = a[:, : m // 2, : k // 2]
            a12 = a[:, : m // 2, k // 2 :]
            a21 = a[:, m // 2 :, : k // 2]
            a22 = a[:, m // 2 :, k // 2 :]
            self.child = Strassen(
                torch.cat(
                    (a11 + a22, a21 + a22, a11, a22, a11 + a12, a21 - a11, a12 - a22)
                ),
                depth - 1,
            )

    def __call__(self, b):
        if not self.depth:
            return torch.bmm(self.a, b)
        k, n = b.shape[-2:]
        n0 = n
        b = torch.nn.functional.pad(b, (0, n % 2))
        n = b.shape[-1]
        b11 = b[:, : k // 2, : n // 2]
        b12 = b[:, : k // 2, n // 2 :]
        b21 = b[:, k // 2 :, : n // 2]
        b22 = b[:, k // 2 :, n // 2 :]
        products = self.child(
            torch.cat((b11 + b22, b11, b12 - b22, b21 - b11, b22, b11 + b12, b21 + b22))
        )
        p1, p2, p3, p4, p5, p6, p7 = products.chunk(7)
        return torch.cat(
            (
                torch.cat((p1 + p4 - p5 + p7, p3 + p5), 2),
                torch.cat((p2 + p4, p1 - p2 + p3 + p6), 2),
            ),
            1,
        )[:, :, :n0]


rows = []
for f, m, k, n in [
    (195, 384, 384, 23),
    (387, 192, 192, 92),
    (6, 192, 192, 5680),
    (387, 96, 96, 181),
    (6, 96, 96, 11360),
    (387, 48, 48, 359),
]:
    a = torch.randn(f, m, k)
    b = torch.randn(f, k, n)
    ref = torch.bmm(a, b)
    for label in ["bmm", "transpose", "strassen1", "strassen2"]:
        if label == "bmm":
            fn = lambda: torch.bmm(a, b)
        elif label == "transpose":
            fn = lambda: (
                torch.bmm(b.transpose(1, 2), a.transpose(1, 2))
                .transpose(1, 2)
                .contiguous()
            )
        else:
            s = Strassen(a, int(label[-1]))
            fn = lambda: s(b)
        times = []
        cpu = []
        for i in range(5):
            t = time.perf_counter()
            c = time.process_time()
            out = fn()
            cpu.append(time.process_time() - c)
            times.append(time.perf_counter() - t)
        row = {
            "shape": [f, m, k, n],
            "method": label,
            "times": times,
            "cpu": cpu,
            "max_abs": float((ref - out).abs().max()),
        }
        rows.append(row)
        Path("/repo/.experiments/probe_bmm.json").write_text(json.dumps(rows, indent=2))
        print(row, flush=True)
