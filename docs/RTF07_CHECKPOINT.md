# RTF 0.7 research checkpoint — 2026-11-11

Work paused at the user's request to commit/push immediately. The 0.7 target is **not achieved**. The previously approved default engine remains opt-in with its original default settings. New settings in this checkpoint are experimental and have not received a fresh listening approval.

All measurements keep quota 7426/10000, cpuset 0–1, 8,000,000,000 bytes and no swap. These are local ARM64 throughput-proxy results, not N1 measurements. No checkpoint or authoritative learned coefficient was changed.

## Full inference candidates

| Report | Experimental options | Warmed median RTF |
| --- | --- | ---: |
| candidate_a1 | head-major KV cache, Cook-Toom phase convolution, initial FFT layout change | 0.943 |
| candidate_a2 | head-major KV cache, Cook-Toom phase convolution | 0.895 |
| candidate_b2 | head-major KV cache, prepacked ARM FFT GEMM, fused convolution epilogues | 0.850 |
| candidate_b3 | b2 plus Cook-Toom phase convolution and packed FFT output layout | 0.849 |

Read the raw reports in `results/rtf07_checkpoint/` for every warmup, measured run, stage, CPU time, waveform comparison and guard result. b2/b3 used three excluded warmups and three measured runs; they are exploratory results, not a final five-run validation. b3 retained RTFs approximately 0.849, 0.831 and 0.861. The first Japanese sentence and reference were unchanged. Second-sentence performance of these candidates has not yet been measured. In-memory/checkpoint hash checks passed for the completed candidates; their full PCM comparison is recorded in each report.

## Changes and rejected approaches

- Added optional prepacked FP32 spectral GEMM using the MIT-licensed ARM 8x12 kernel, with native input packing and product unpacking. Preserve `ARM_COMPUTE_LICENSE` and embedded source notices.
- Added optional FFT overlap-save crop/bias/activation/residual fusion and Winograd epilogue fusion. Residuals with non-contiguous channel rows are supported.
- Combined two FFT input padding copies into one equivalent padding operation.
- Added optional head-major AR KV caches, exact-value GEMV16 prototype and Cook-Toom convolution/upsampling prototypes. Original weights remain authoritative.
- Added build-time source-change detection to prevent a manifest from silently describing sources modified during compilation.
- Generic Strassen BMM, replacing single-token attention, broader Cook-Toom residual transforms, oneDNN prefill linear layers, and global FFT-size tuning did not provide a convincing overall improvement. They remain research options/probes, not defaults.

`candidate_b1` failed a residual-layout check; it was retained and corrected before b2. The first fusion test harness accidentally mutated its reference tensor; the corrected harness passed. Native fusion/layout validation passed 75 cases before the final input-pack vectorization. Earlier checkpoint test coverage also includes 24 host tests, but the complete host suite was not rerun after the final edits in this commit.

## Latest edit and resumption

After b3, `fft_pack` was vectorized with NEON 4x4 transposes. That latest input-pack edit has **not** been measured in full inference. A rebuild was started locally before the user requested an immediate checkpoint; do not assume an old binary matches the current sources. The pre-vectorization build manifest is retained as provenance. Rebuild the current source with `build_cpufast_ar.py` before using it.

Continue by validating the latest native input packing against the previous tensor layout, rerunning native fusion tests and host tests, then measuring baseline/candidates back-to-back. Retain cold starts and slow samples. Validate the second sentence and obtain listening review before adopting a changed candidate. Keep the CPU quota frozen.

The `experiments/rtf07/` scripts document the probes and use container paths `/repo`, `/work`, and `/baseline`. They expect the same separately supplied model assets, full HuBERT, Japanese resources, private reference and, for layer probes, `layer_captures.pt`. Those assets and the local Docker image are not distributed. Scripts write scratch outputs under `/repo/.experiments`; create that directory first. No generated audio, model weights or compiled binaries are published.
