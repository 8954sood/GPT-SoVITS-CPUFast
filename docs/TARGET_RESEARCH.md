# Actual 2-vCPU ARM server: next hypotheses

The target is the configured ARM server under its 2-vCPU/8,000,000,000-byte Docker limit. The immediate objective is a warmed end-to-end RTF below 1 with unchanged authoritative model weights, value-preserving coefficient caches and FP32 arithmetic. Local 0.7426-CPU measurements are not the target-server score. Keep model loading and every warmup/slow run explicit; compare equal reference, text, seed, semantic tokens, duration and audio.

## Evidence consulted

- GPT-SoVITS V5's phase upsampling and channels-last layout reduced decoder cost, but much less of total synthesis; full XNNPACK and residual-branch parallelization were not consistently helpful. Both phase upsampling and packed FP32 layouts already exist here. V5's ORT FFN gain was in a different CFM model and hardware proxy, so it does not establish an AR prefill gain here.
- Audio8's channel-selected direct tap GEMM improved its decoder, whereas larger Winograd variants won some microbenchmarks but lost on full TTS. The current `tap_small` path checks <=48 channels. Probe 96/192 channel geometries on the actual server before broadening it. Local proxy probes made these larger tap cases much slower than FFT; server results may differ.
- Audio8's grouped-query attention shares K/V among query heads. This GPT-SoVITS AR model has distinct per-head K/V, so the same grouping is not automatically applicable. It still suggests testing thread counts and head-major cache layout without altering attention semantics.
- [Supertonic 3](https://github.com/8954sood/supertonic3-optimization) found reference conditioning worth caching, but this CPUFast checkout already caches `decode_ge`, `decode_ge_text`, prompt semantics and reference features. Avoid claiming those existing caches as a new gain. Its 2-vCPU N1 result is a different model and cannot predict this model's RTF.
- Prior local probes found the custom single-token attention and most generalized Winograd/Strassen cases slower. Do not reintroduce them without target-hardware evidence. W8A8 and FP16 arithmetic remain outside this task's earlier FP32/lossless coefficient constraint even though another remote branch added them as separate opt-in experiments.

## Prepared diagnostics

1. Record a separate profiled inference using `benchmark_cpufast.py --profile-output PATH`; this extra invocation is excluded from RTF statistics. CPU operator totals can overlap across threads and contain profiler overhead.
2. Run `experiments/rtf07/bench_ar_gemv.py` to compare native GEMV tile 8/16 and one/two threads across the actual AR matrix shapes. It checks exact output equality and reports every measured call. Synthetic tensors make it a kernel probe, not a quality or full-RTF result.
3. Run `experiments/rtf07/bench_decoder_methods.py` to compare packed FFT with tap GEMM on representative decoder shapes. Synthetic weights, differing cache behavior, and isolated calls mean any apparent win still requires full inference and listening.
4. Test the smallest promising change on both sentences, with a fresh approved-waveform comparison and independent warmed repeats. Do not select from a single lowest measurement.

The probes and separate profiling invocation completed on the target in [run 43129199602](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43129199602); raw data is in `results/n1_diagnostics_20261112`. The tap synthesis median RTF was 1.5684 (three samples). Two threads beat one for the main AR GEMV shapes; tile 16 showed no consistent gain versus tile 8. Direct tap was slower than packed FFT for every tested 96/192-channel decoder geometry. Native attention was faster at lengths 64 and 128 with two threads but slower at 256 and 512, motivating a short-context-only candidate. Synthetic timings do not establish end-to-end speed or speech quality. Profiling self CPU times include overhead and should not be summed into wall time. Sources, results and implementation from the referenced projects are evidence and must not be modified by this repository's workflow.

## Short-context attention result

[Run 43129998418](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43129998418) compared the tap baseline with native attention only at cache lengths up to 128. Tap/attention median RTF was 1.5546/1.5734 for sentence 1 and 1.5737/1.5819 for sentence 2. The semantic and PCM outputs were exactly equal. Shape-aware profiling shows prefill already has length 272, so this threshold never activates on this workload; the timing differences do not measure the candidate. The longer-context microbenchmark showed the native path slower, so do not adopt it for the target. Full JSON and guards are in `results/n1_attention_20261112`.

## Small-channel decoder probe

[Run 43130554325](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43130554325) compared tap, FFT and channels-last FP32 convolution on synthetic 24/48-channel geometries. At 48 channels, kernel 7, dilation 1, medians were 0.0462s tap and 0.0392s channels-last; other tested geometries did not show a compelling improvement. A geometry-specific opt-in full synthesis candidate is necessary before any adoption. Raw values and guard are in `results/n1_small_20261112`.

## Thread wait policy

On the actual 2-vCPU host, [run 43131083453](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43131083453) measured ACTIVE/GOMP_SPINCOUNT=100000 at median RTF 1.5733 and PASSIVE/GOMP_SPINCOUNT=0 at 1.6206, three samples per policy. PCM and semantic output were exactly equal. Keep ACTIVE; the earlier V5 proxy result does not transfer to this host. Results and guards are in `results/n1_scheduling_20261112`.

## 48-channel packed full synthesis

[Run 43131616992](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43131616992) selected four measured 48-channel convolutions and compared full TTS. Tap/candidate median RTFs were 1.5668/1.5887 and 1.5932/1.6037 for sentences 1/2. Semantics match; PCM differs by at most one unit (11 and 21 samples respectively). No overall gain, candidate not adopted or listening approved. Raw data under `results/n1_packed48_20261112`; output files use `attention` as a historical generic candidate label.

## FFT size probe

On [run 43131935673](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43131935673), 384 channels, kernel 11, dilation 5 favored FFT 256 (0.1427s) over the current 128 (0.1887s), while 96 channels, kernel 11, dilation 5 favored 512 (0.0725s) over 256 (0.0810s). Other shapes and all measurements are in `results/n1_fft_20261112`. A two-geometry selective full-synthesis comparison is required before adoption.

## FFT selection and AR cache

[Run 43133005529](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43133005529) compared the two selected FFT sizes against tap. Tap/candidate medians were 1.5766/1.5911 and 1.5953/1.5709 across two sentences; mixed and noisy, so do not adopt based on these three samples. Semantic output matched; PCM changed by at most one unit. In the same run, exact-FP16 versus FP32 storage of the same coefficients produced bit-identical GEMV outputs. At two threads the FP32 cache improved all five synthetic matrix shapes by approximately 14–27%, motivating a full synthesis test. All arithmetic remains FP32; higher memory use and full-model performance remain to be checked. Results: `results/n1_fftcandidate_20261112`.

## FP32 AR cache full synthesis

[Run 43134001131](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43134001131) measured tap/FP32-cache RTF 1.5842/1.6892 and 1.5922/1.6967 on two sentences. AR time rose from about 4.09s to 4.70s for sentence 1, and output was PCM-exact. Target guard peaks were about 5.45GB. The cache-hot single-matrix probe did not predict full-model performance; do not adopt. Results: `results/n1_arcache_20261112`. Next probe traverses a synthetic 24-layer weight set to reflect model-scale cache pressure.

## Model-sized AR prefetch probe

[Run 43135138734](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43135138734) traversed 151MB of synthetic 24-layer AR weights. Prefetch distances 0/64/128/256 gave medians 11.05/10.37/9.71/9.02ms per synthetic token, with bit-exact output across all 96 matrices. Distance 256 warrants end-to-end testing; the traversal does not model attention, layer computation or full inference. Data and guard: `results/n1_prefetch_20261112`. Initial run 43134968081 selected the previous commit due to dispatch immediately after push and failed input validation before server access; rerun used the intended commit.
