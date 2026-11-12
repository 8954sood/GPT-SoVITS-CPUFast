"""Run CPU-only host training with observation, without CPU/RAM enforcement."""

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A training command is required")
    if args.report.exists():
        parser.error("Choose a new report path to preserve prior runs")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")
    worker = subprocess.Popen(command, env=env)
    process = psutil.Process(worker.pid)
    started = time.perf_counter()
    peak = cpu_seconds = 0
    status = "running"
    try:
        while worker.poll() is None:
            try:
                tree = [process] + process.children(recursive=True)
                rss = 0
                cpu = 0
                for child in tree:
                    try:
                        rss += child.memory_info().rss
                        timing = child.cpu_times()
                        cpu += timing.user + timing.system
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                cpu_seconds = max(cpu_seconds, cpu)
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.25)
    except KeyboardInterrupt:
        status = "interrupted"
        worker.terminate()
    finally:
        code = worker.wait()
        elapsed = time.perf_counter() - started
        report = {
            "command": command,
            "platform": platform.system(),
            "machine": platform.machine(),
            "device_policy": "CPU only; train script rejects GPU/MPS parameters",
            "cpu_quota_cores": None,
            "memory_limit_bytes": None,
            "scope": "unrestricted host training observation; not an inference RTF benchmark",
            "logical_cpus": os.cpu_count(),
            "physical_ram_bytes": psutil.virtual_memory().total,
            "peak_sampled_tree_rss_bytes": peak,
            "cpu_seconds": cpu_seconds,
            "elapsed_seconds": elapsed,
            "average_cpu_cores": cpu_seconds / elapsed,
            "exit_code": code,
            "status": status
            if status == "interrupted"
            else ("complete" if code == 0 else "failed"),
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
