"""Run a serial CPU experiment under a CPU-time budget and RSS watchdog.

macOS has no cgroup CPU affinity/RSS controller. This enforces an aggregate
CPU-time quota with SIGSTOP/CONT and aborts on sampled process-tree RSS excess.
It is not a kernel-enforced 2-core cpuset or a hard address-space limit.
"""

import argparse
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=float, default=2)
    parser.add_argument("--memory-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.cores <= 0 or args.memory_bytes <= 0:
        parser.error("A command and positive resource limits are required")
    env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = str(max(1, int(args.cores)))
    env["TOKENIZERS_PARALLELISM"] = "false"
    worker = subprocess.Popen(command, env=env, start_new_session=True)
    root = psutil.Process(worker.pid)
    peak_rss = peak_debt = cpu_seconds = stopped = 0.0
    previous_cpu = 0.0
    started = previous_wall = time.monotonic()
    debt = 0.0
    violation = None
    samples = 0
    try:
        while worker.poll() is None:
            now = time.monotonic()
            rss = cpu = 0.0
            try:
                processes = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            for process in processes:
                try:
                    rss += process.memory_info().rss
                    times = process.cpu_times()
                    cpu += times.user + times.system
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            samples += 1
            peak_rss = max(peak_rss, rss)
            if rss > args.memory_bytes:
                violation = (
                    f"Process-tree RSS {int(rss)} exceeds {args.memory_bytes} bytes"
                )
                os.killpg(worker.pid, signal.SIGKILL)
                break
            cpu_seconds = max(cpu_seconds, cpu)
            debt = max(
                -0.04,
                debt + max(0, cpu - previous_cpu) - args.cores * (now - previous_wall),
            )
            peak_debt = max(peak_debt, debt)
            previous_cpu, previous_wall = cpu, now
            if debt > 0.02:
                delay = min(debt / args.cores, 0.05)
                os.killpg(worker.pid, signal.SIGSTOP)
                try:
                    time.sleep(delay)
                finally:
                    os.killpg(worker.pid, signal.SIGCONT)
                stopped += delay
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        os.killpg(worker.pid, signal.SIGCONT)
        os.killpg(worker.pid, signal.SIGTERM)
        violation = "Interrupted"
    except ProcessLookupError:
        pass
    finally:
        code = worker.wait()
        elapsed = time.monotonic() - started
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        kernel_peak = usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
        if kernel_peak > args.memory_bytes and violation is None:
            violation = f"Kernel-reported child peak RSS {kernel_peak} exceeds {args.memory_bytes} bytes"
        cpu_seconds = max(cpu_seconds, usage.ru_utime + usage.ru_stime)
        report = {
            "command": command,
            "cpu_quota_cores": args.cores,
            "memory_limit_bytes": args.memory_bytes,
            "peak_sampled_tree_rss_bytes": int(peak_rss),
            "kernel_child_peak_rss_bytes": kernel_peak,
            "cpu_seconds": cpu_seconds,
            "elapsed_seconds": elapsed,
            "average_cpu_cores": cpu_seconds / elapsed,
            "quota_stop_seconds": stopped,
            "maximum_cpu_debt_seconds": peak_debt,
            "samples": samples,
            "memory_control": "sampled process-tree RSS watchdog plus final kernel child peak check; not a kernel hard cap",
            "cpu_control": "aggregate CPU time quota; 20ms CPU-time burst allowance; not cpuset",
            "violation": violation,
            "exit_code": code,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    sys.exit(1 if violation else code)


if __name__ == "__main__":
    main()
