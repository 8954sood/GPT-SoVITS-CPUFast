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

## Full synthesis with AR prefetch

[Run 43135656821](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43135656821) compared the approved tap path to prefetch distance 256. Median RTF improved 1.5717 to 1.5274 and 1.5879 to 1.5529 on the two sentences (three samples each). AR median fell 4.044 to 3.742s and 3.679 to 3.484s. Both outputs were PCM-exact and weights unchanged. This is provisional pending independent five-sample confirmation; reports and guards are in `results/n1_prefetchcandidate_20261112`.

Independent five-sample verification [run 43136472254](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43136472254) reproduced the result: tap/prefetch median RTF 1.5787/1.5394 and 1.5870/1.5552. Every candidate output was PCM-exact, checkpoints/parameters unchanged and guards passed. Sentence 2 candidate included a 1.8727 outlier; all five runs, including that outlier, remain in `results/n1_prefetchrepeat_20261112`. The 256 setting is now the default for the FP32 engine, with `ar_prefetch=0` available for the prior path. This remains above the target RTF.

[Run 43136630015](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43136630015) extended the synthetic prefetch sweep: 192/256/320/384 all measured approximately 9.02–9.05ms per synthetic token, so there is no compelling reason to replace the independently verified 256. Details are in `results/n1_prefetch_extended_20261112`.

## Three-tap decoder probe

[Run 43137924850](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43137924850) compared native Winograd against channels-last direct FP32 convolution at 15 synthetic geometries. Direct convolution favored 96/48/24 channels with dilation 3 or 5, while Winograd remained faster for 192/384 channels. Synthetic timings are in `results/n1_winograd_20261112`; selective end-to-end testing and listening remain necessary.

The subsequent full-model [run 43138450302](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43138450302) selected six direct three-tap convolutions with the verified prefetch enabled for both variants. Baseline/candidate RTF medians were 1.5407/1.5429 and 1.5661/1.5670; the candidate did not improve end-to-end speed and was not adopted or listening approved. PCM differed by at most one unit, with 17/21 changed samples. Raw runs and guards: `results/n1_direct3_20261112`.

## Prefetch locality

[Run 43139552360](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43139552360) measured locality hints 0/1/2/3 at 9.288/9.006/8.982/8.949ms per synthetic model-sized traversal. Hint 0 had two slow outliers; all samples are preserved. Hints 1–3 differ by less than one percent, so keep the independently validated locality 1. Results: `results/n1_locality_20261112`.


## Streaming FFT products

[Run 43154054097](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43154054097)
compared the prefetch-256 FP32 tap baseline against four-frequency streamed
spectral GEMM/complex reconstruction. It removes the full three-plane product
buffer while retaining the existing ACL kernel and FP32 evaluation order.
Baseline/candidate medians were **1.5336/1.5151** and **1.5410/1.5183**, with
three excluded warmups and three retained samples per variant. Decoder medians
were 4.990/4.897s and 4.407/4.388s; AR timing varied despite unchanged AR code.
The small full-RTF improvement is provisional and needs independent five-sample
verification before adoption. Every measured candidate PCM and semantic output
matched its baseline exactly, parameter/checkpoint hashes were unchanged, and
guards passed. Synthetic packed-GEMM/reconstruction improved at 192/96 channels
but was approximately tied at 384 channels. Raw reports and provenance are in
`results/n1_fftstream_20261112`; local Docker results were correctness checks only.


## Channel-parallel transform and finishing

[Run 43154873191](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43154873191)
compared the original FP32 prefetch-256 tap profile against channel-parallel
Winograd input/output and FFT finishing. This retains each channel's arithmetic
and distributes independent channels across the existing two Torch threads.
Baseline/candidate RTF medians were **1.5555/1.5212** and **1.5321/1.5157**,
three warmups and three retained samples per case. Decoder median fell
5.135/4.870s and 4.515/4.220s, while unchanged AR timing rose by about 0.06/0.13s.
All measured candidate PCM/semantic outputs were exact, parameter/checkpoint
hashes unchanged and guards passed. Independent five-sample verification is
required before adopting this small overall gain. Raw reports and provenance:
`results/n1_transforms_20261112`. The FFT streaming flag was off in both cases.


Independent five-sample stream confirmation
[run 43155674421](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43155674421)
reproduced the small gain: baseline/stream medians **1.5411/1.5248** and
**1.5598/1.5446**. Decoder medians were 5.083/4.942s and 4.549/4.427s, with
slightly slower AR despite unchanged AR code. All ten measured candidate
waveforms were PCM-exact, semantic tokens/durations unchanged, and integrity
checks/guards passed. Keep the streaming implementation as an opt-in; this
roughly one-percent total gain is specific to the two target workloads and
does not reach real-time synthesis. Raw runs, including every warmup and slow
sample, are in `results/n1_fftstream_repeat_20261112`.

## Prepared FP32-cache prefetch for future student validation

Trained student coefficients may no longer round-trip exactly through FP16
storage. The existing explicit FP32 AR cache preserves them, but the N1
full-model experiment above found that a larger FP32 cache can lose despite
winning cache-hot matrix probes. Its native path previously hard-coded
prefetch 0. A new opt-in `lossless_linear_f32_prefetch` uses the existing FP32
GEMV arithmetic and coefficient storage with explicit distance 64/128/256.
`optimize(ar_fp32_cache=True)` keeps prefetch 0 when unspecified; requesting a
nonzero distance is an experiment, not a recommended student default.

The Linux ARM64 local correctness test compares all five actual AR matrix
shapes using FP32 coefficients that cannot be exactly converted to half, at
one/two threads and four distances. Every output was bit-exact to the existing
no-prefetch FP32 operator, original coefficients remained unchanged, and the
TorchScript classes imported/compiled successfully. These are tiny correctness
checks, not performance evidence. No student or full-model N1 run has validated
this option. Keep original and trained student checkpoints separate and obtain
quality approval before measuring a student on the actual server.


## Independent runtime and combined original-model validation

[Run 43156673437](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43156673437)
used the unchanged original model on the real N1 with 2 CPU, 8,000,000,000 bytes
and no swap. Each of eight fresh processes excluded three warmups and retained
five samples. Non-baseline variant order was reversed for sentence 2.

| Original FP32 configuration | Sentence 1 median RTF | Sentence 2 median RTF |
| --- | ---: | ---: |
| Approved tap/prefetch256 baseline | 1.5340 | 1.5501 |
| Channel-parallel transforms only | 1.4967 | 1.4956 |
| Inference context only | 1.5107 | 1.5171 |
| FFT streaming + parallel transforms + inference context | **1.4732** | **1.4464** |

The combined improvement of 3.97%/6.69% is directly measured, not a sum of
isolated gains. Decoder medians fell 5.053 to 4.786s and 4.492 to 4.188s.
Every one of the 30 measured candidate outputs was PCM-exact to its baseline;
semantic tokens, duration, sample rate, reference, request/seed and resource
settings matched. All checkpoint/parameter checks and guards passed, with
peak sampled tree RSS below 5.4GB. Every warmup and measured run, including
slower observations, is in `results/n1_runtime_20261112` with provenance.

Keep these original-model paths as opt-in improvements. The default engine
configuration and original weights remain unchanged. This does not reach RTF
1, does not establish an optimum or coverage of other voices/languages, and
does not predict the performance/quality of separately trained students.
See `ARM_BENCHMARK.md` for the exact original-model settings.

## Approved G: trained FP32-cache actual N1 validation

[Run 43161518437](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43161518437)
fixed source `e32a7096a3221d3a7682b700e949d33d0b9dec81` and the private,
SHA-verified approved G bundle. Model bytes are excluded from Git and artifacts.
The user approved Korean listening only. Three warmups preceded each case;
the sweep and JA/EN had three samples, selected Korean combined had five.

| Korean configuration | Median RTF | Wall seconds | AR seconds | Decoder seconds |
| --- | ---: | ---: | ---: | ---: |
| Original combined, exact-half coefficient cache | 1.4440 | 8.346 | 3.409 | 4.542 |
| G, full FP32 cache, prefetch 0 | 1.4709 | 8.678 | 3.901 | 4.383 |
| G, prefetch 64 | 1.4779 | 8.719 | 3.942 | 4.364 |
| G, prefetch 128 | 1.4786 | 8.724 | 3.870 | 4.437 |
| G, prefetch 256 | 1.5155 | 8.941 | 4.073 | 4.470 |
| G combined, prefetch 0, independent five samples | 1.4414 | 8.504 | 3.821 | 4.237 |

Retain prefetch 0. Positive hints did not improve this workload. Model changes
altered generated duration from 5.78 to 5.90s: the nearly equal Korean RTF does
not mean lower latency. The trained AR's FP32 coefficient cache plausibly offsets
the reduced layer count: AR became slower despite fewer layers, while decoder
time fell. This is evidence about this complete model, not a proof of a pure
memory bottleneck. No individual optimization gains are added together.

JA original/G combined measured RTF 1.4688/1.4033, but wall time increased
8.784/9.233s with 5.98/6.58s outputs. EN measured 1.4884/1.4300 and wall time
7.293/6.550s, output 4.90/4.58s. These are one-text performance smoke checks,
not listening approval or a language-wide speed claim.

Every within-N1 Korean prefetch/combined PCM and token sequence was exact to
G0. All three generated semantic hashes matched the local approved render;
cross-platform PCM hashes differed and are reported explicitly. Asset,
parameter, checkpoint, resource checks passed. Peak tree RSS was below 5.48GB
for timed jobs and 6.41GB for the separate profile.

The extra unshaped CPU profile, excluded from RTF, recorded self operator times
of 1.772s FP32 GEMV, 1.498s addmm, .959s FFT stream, .650s bmm, and .529s
CPU flash attention. Threaded CPU totals overlap and include profiling overhead;
unshaped addmm totals cannot all be attributed to the classifier. A concrete
next candidate caches a failed predictor half-storage eligibility check, while
retaining the same FP32 F.linear arithmetic. It requires its own N1 comparison.
Raw JSON, guards, profile and provenance: `results/n1_student_g_20261112`.

### Failed G predictor eligibility-cache speed hypothesis

[Run 43163928379](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43163928379)
fixed source `2c94b384a6b5f55cfdf2709221c620374c24d631`. The approved G
combined runtime, FP32 cache0, compared the existing failed-eligibility check
against an opt-in negative cache. Three excluded warmups and five retained
samples per configuration measured **RTF 1.400229/1.404697** (+0.319%, slower).
Wall medians were 8.261/8.288s; CPU 14.260/14.159s; AR 3.767/3.745s; decoder
4.142/4.118s, same 5.90s output. Small stage/CPU reductions do not establish
an overall latency gain. Do not enable the negative cache in the next model
comparison or describe it as a successful optimization.

All five measured candidate PCM/token/duration outputs matched the baseline
exactly. Original/model parameter, checkpoint, student asset and resource checks
passed. Peak tree RSS was 5.32GB (6.38GB in the separate profile). Every slow
sample, guard, profiler output and provenance is retained in
`results/n1_student_predictor_20261112`. This chronological A/B run does not
establish a sub-percent effect or a server-wide optimum.

The rejected predictor negative-cache runtime/API/workflow paths were then
removed. Its exact source remains reproducible at the run's commit above;
raw failure reports are retained. Original NativePredictor behavior is restored.

## Approved H2/H3: actual N1 inference-profile gain

[Run 43164950232](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43164950232)
fixed source `023439e23d12508e9cabb7f672d627982e197ee6`, three excluded
warmups and five retained Korean samples per configuration. Both models have
18 AR layers and the same approved G 13-branch decoder. H3 is a separately
user-approved inference copy with 73 AR matrices rounded to exactly
FP16-representable values; biases, normalization, embeddings and decoder remain
unchanged. Original H2 and teacher checkpoint bytes are preserved.

| Configuration | Median RTF | Wall s | CPU s | AR s | Decoder s | Audio s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2 FP32 cache0 | 1.3623 | 6.839 | 11.764 | 2.871 | 3.542 | 5.02 |
| Same H3 weights, FP32 cache0 control | 1.3589 | 6.931 | 11.836 | 3.010 | 3.567 | 5.10 |
| H3 exact-half cache256, FP32 arithmetic | **1.2313** | **6.280** | **10.693** | **2.350** | 3.607 | 5.10 |

The same-H3 cache/prefetch execution profile improved measured full latency/RTF
by **9.39%**. This jointly changes coefficient storage and prefetch 0→256;
it does not isolate storage alone. All five half-cache candidate PCM outputs,
semantic tokens and durations were exactly equal to the H3 FP32-cache control.
H2/H3 generate different tokens and lengths, so their RTF improvement of 9.61%
and wall-time improvement of 8.17% are separate cross-model observations.
Every variant's generated semantic hash matched its own local approved render;
cross-platform local/N1 PCM hashes differed, as in G. Approval covers only the
same Korean reference/text/seed, not other languages or general zero-shot use.

The unchanged decoder now consumes about 57% of H3 wall time: 3.607s, or
.7072 decoder seconds per audio second, versus AR .4608 seconds per audio second.
The separate excluded profile measured classifier range 121 calls, input
1×512, weight 1025×512, only **35.775ms total CPU**. The 1.209s addmm total
therefore does not identify classifier projection as the bottleneck. Other
profile self times were FFT stream .875s, half-cache GEMV .797s, bmm .547s,
CPU attention .394s. Totals include overhead/overlap and are not added to wall time.

All asset, parameter, checkpoint and fixed 2CPU/8GB/no-swap guards passed.
Peak sampled tree RSS was under 5.16GB (5.86GB with profiler). Preserve this
approved H3 cache256 profile for subsequent isolated decoder comparisons.
RTF remains above 1; no real-time or 0.7 result is claimed. Full raw data,
all slow samples, guards, profile and provenance:
`results/n1_student_h2_h3_20261112`.

## Approved K decoder and rejected temporal tiling

[Run 43166415026](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43166415026)
fixed source `b5f45da8613d796edc9ae8e8acde311000edd920`. H3 and separately
approved K share identical AR checkpoint bytes and half-cache256 execution;
K removes the first decoder stage's original 3-tap branch and has 100 CPU
recovery updates. Each excludes three warmups and retains five Korean samples.
Both produce the same 120 generated tokens and 5.10s output.

| Configuration | Median RTF | Wall s | CPU s | AR s | Decoder s |
| --- | ---: | ---: | ---: | ---: | ---: |
| H3 decoder13 | 1.2461 | 6.355 | 10.769 | 2.287 | 3.598 |
| K decoder12 | 1.2265 | 6.255 | 10.636 | 2.341 | 3.547 |

This measured total improvement is only 1.57%; decoder time fell 1.42% while
AR rose. Treat it as a small, unreplicated observation, not a confirmed large
gain or a generic model-size speed rule. Different decoder coefficients change
waveforms and had separate Korean listening approval; cross-model PCM equality
is not required. Model/resource guards passed with timed RSS below 5.03GB.

The excluded K profile verified its instrumented PCM/token/duration exactly
against the same process's unprofiled result. Actual tap input lengths were
76800 at 48 channels and 153600 at 24 channels. Those four selected geometries
then ran synthetic full-sequence versus temporal tile 1024/4096/8192, each
three warmups and five retained samples. All sixteen outputs were FP32 exact.

| Tap geometry | Full sequence s | Tile1024 s | Tile4096 s | Tile8192 s |
| --- | ---: | ---: | ---: | ---: |
| C48 K7 D1 | **.03836** | .04877 | .04200 | .04027 |
| C48 K11 D5 | **.05925** | .07718 | .06360 | .06122 |
| C24 K7 D1 | **.02250** | .04206 | .02797 | .02532 |
| C24 K11 D5 | **.03656** | .06556 | .04298 | .03941 |

Reject temporal tiling: every candidate was slower, including CPU time.
No production TapConv change or full-TTS tiling job is justified. New probes
must change the actual computation strategy rather than repeat these tiles.
The separate profile's complete tap ranges total 1.0660 CPU seconds; the four
most costly categories are C48/C24 × K7/K11, dilation1, at .7343s combined.
These inclusive diagnostic ranges contain overhead and are not added to wall
time or other operator totals. Raw timings, all slow samples, numerical errors,
actual shapes, guards, profile and provenance:
`results/n1_student_k_20261112`.

## Approved L: isolated approximate AR runtime

[Run 43168034387](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43168034387)
fixed source `a84f4b90cc1167fcc7e27f4552a45e48043d6c8a`, three warmups and
five retained Korean samples each. It reused the identical private K checkpoint
bundle. L changes only derived AR inference to W8A8 projections/native FP32
attention; its normal acoustic engine remains FP32, with zero `fast_half`
modules and mixed global fast mode false. It had separate Korean listening
approval and does not rewrite original coefficients/checkpoints.

| Configuration | Median RTF | Wall s | CPU s | AR s | Decoder s | Audio s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K half-cache256 | 1.2276 | 6.261 | 10.630 | 2.264 | 3.534 | 5.10 |
| L AR-only W8A8/native FP32 attention | **1.1414** | **5.730** | **9.771** | **1.925** | 3.424 | 5.02 |

The measured RTF gain is 7.02%, wall-time gain 8.48%; tokens/duration change
from 120/5.10s to 118/5.02s. Every L generated semantic hash matches the local
approved L render. The underlying K semantic comparison is false as expected;
the separate runtime-approved match is explicitly true. Cross-platform PCM
differs. All model/reference/checkpoint/asset guards passed; timed RSS was
below 5.06GB, separate profiler 5.71GB. Actual AR/classifier and acoustic tap
profile ranges observed the global Torch intra-op pool at two threads; legacy
`native_workers=1` metadata is not a thread limit.

The excluded L profile recorded self times of 1.310s addmm, .775s FFT stream,
.442s bmm, .433s INT8 projection and .413s native head attention. The decoder
is now about 60% of wall time. Profiled PCM/tokens/duration matched the same
process unprofiled output exactly. Preserve L as an explicit approved candidate;
**RTF remains above 1**, so the conditional independent sub-real-time
confirmation was not run. Raw data: `results/n1_student_l_20261112`.

## Rejected explicit im2col tap probe

[Run 43169070226](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43169070226)
fixed source `faa35c07a7135b88267223d3b89ae2f764c00820`. Its four synthetic
bulk geometries came from the actual L profile, C48 length75520 and C24
length151040, dilation1 and kernels7/11. Three warmups and five measurements
included actual materialization for every full/tiled im2col candidate.

| Geometry | Existing taps s | Best im2col s | Best setting |
| --- | ---: | ---: | --- |
| C48 K7 | **.03719** | .06701 | Full sequence |
| C48 K11 | **.05699** | .10784 | Full sequence |
| C24 K7 | **.02264** | .05693 | Tile8192 |
| C24 K11 | **.03468** | .07957 | Full sequence |

All im2col candidates were slower. Different accumulation order changed FP32
values by at most 8.6e-6 to 1.9e-5 across geometries; outputs remained finite,
coefficients unchanged, and guard peak RSS was 577.6MB. Reject the approach:
no production tap implementation or full-TTS job was added. Every measured
candidate, warmup, slow sample, error, memory estimate and guard is retained in
`results/n1_im2col_tap_20261112`.

The next prepared attention experiment adds explicit
`cpufast_fast_ar.build(..., head_major=True, torch_head_attention=True)`.
Default false preserves L's native attention; true keeps the same W8A8
projections and head-major KV layout while using Torch SDPA. Tiny Linux ARM
one/two-thread block checks passed mask/all-masked-head, cache-write identity
and FP32 output tolerance. This preparation has no full-TTS quality or actual
N1 speed claim; new listening approval and target measurement are required.

## M attention-only comparison: no total speed gain

[Run 43170346259](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43170346259)
fixed source `b6dbc40adc1d2fda8296994eeb6ebe0c000c8604`, same K coefficients,
W8A8 projections, head-major KV cache, FP32 acoustics, Korean reference/text/seed.
Each case excluded three warmups and retained five samples. M changed only the
native FP32 attention implementation to Torch SDPA.

| Runtime | Median RTF | Wall s | CPU s | AR s | Decoder s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Approved L native attention | **1.1592** | **5.8190** | 9.8961 | 2.0536 | 3.3696 |
| M Torch SDPA attention | 1.1609 | 5.8276 | 9.8438 | 2.0356 | 3.3322 |

Both generate 118 tokens and 5.02s, and every one of the five M PCM/token/duration
outputs exactly matched N1 L. M inherited Korean approval from exact local L
PCM equality; it did not have a separate user approval answer. JA/EN are outside
the approval. Guards and model/reference/checkpoint hashes passed. Profiled M
also matched its unprofiled same-process waveform exactly.

The overall RTF difference is +.148%, not an improvement. Small stage/CPU
reductions do not establish full synthesis gain; all slow samples, including
the baseline outlier, remain in `results/n1_student_m_20261112`. The excluded
M profile recorded self times of .377s Torch CPU attention, .517s INT8 projection,
.828s FFT stream and 1.219s addmm. Do not add these to wall time or compare
cross-run profiling numbers as an isolated speed estimate.

Keep L native attention as the operational baseline. Remove M benchmark/CLI
and workflow selection after preserving the run SHA/raw data. The default-false
`cpufast_fast_ar` attention API remains for Main's archived research helper,
per explicit request. No redundant M performance rerun is justified.

## Narrow decoder execution support, quality gate still separate

The rejected N640 voice candidate is not deployed or benchmarked on N1.
Generic execution support remains available for separately approved future
channel selections: immutable `decoder_channel_keep` IDs must match manifest,
spec and trained decoder spec before loading. An inline whole-bundle approval
may declare `ar_runtime: int8_native`, authoritative `ar_cache_storage:
exact_fp16`, and source-pinned `runtime_implementation`. Checksum/path checks
reject changed implementations and traversal. Legacy manifests retain their
previous behavior; the K bundle still uses the separate approved L runtime.

`studentdecoderint8` compares two approved bundles with identical AR bytes/cache,
W8A8/native attention on both, and fixed semantic tokens/durations. A selected
channel model requests the opt-in zero-padded FFT output cache and reports
actual padded modules; cross-model decoder PCM equality is not required.
No new-width server job runs before its listening approval and immutable assets.

Local Linux ARM correctness verified output-cache padding at 320/160/80/40/20
channels bit-exact to explicit zero-padded references, including one/two threads,
batch1/2, layout/stream, activation/residual, parameter identity and immutability.
The prospective 672-width model's C42/C21 three-tap Winograd path also passed
dilation1/3/5, ragged time137, noncontiguous residual, batch1/2, direct-convolution
FP32 tolerance and bit-exact serial/parallel comparisons. Its spatial SIMD does
not impose a four-channel count requirement. These are compatibility checks,
not speech-quality or actual-N1 speed claims.

## P: approved FFT FP16 products, first actual N1 pair

Run `43174364747`, source `6c4ec57d9325ac86b725fe68d1afdeaa325fd00f`,
completed on the fixed N1 2-vCPU / 8,000,000,000-byte / no-swap target.
Both cases used three excluded warmups and five measured samples, identical K
checkpoint bytes, L's W8A8/native FP32 AR, reference, Korean text and seed42.
P had separate explicit Korean listening approval. JA/EN are not approved.

| Runtime | Median RTF | Wall s | CPU s | AR s | Decoder s | Peak sampled tree RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L, FP32 streaming FFT products | 1.161729 | 5.831880 | 9.916521 | 2.075922 | 3.400078 | 5.059 |
| P, materialized FP16 FFT products | 1.133439 | 5.689864 | 9.629696 | 1.922116 | 3.371940 | 6.371 |

P converts packed spectral inputs and derived spectral coefficient caches to
FP16 and accumulates products in FP16, returning materialized FP32 products.
Forward/inverse transforms, Tap, Winograd and finishing remain FP32. Only the
30 large FFT modules change; global mixed fast mode is disabled. Learned
parameters/checkpoint bytes remain unchanged. Streaming is disabled, so the
comparison includes its additional products allocation and unpack cost.

Overall RTF decreases 2.44%, but the unchanged AR stage also decreases 7.4%;
the changed decoder decreases only .83%. This is not yet convincing isolated
decoder gain or a sub-1 result. Keep L as the operational baseline pending
further evidence. All five outputs retain the same 118 semantic tokens and
5.02s duration. The candidate's separate profile matches its same-process
unprofiled PCM/token/duration exactly. All resource and file/parameter guards
pass; all warmups and slow samples remain in `results/n1_student_p_20261112`.

The excluded P profile records self times of .5012s `fft_hgemm`, .1856s
`fft_unpack_layout`, .1687s `fft_pack` and .2173s `copy_`. These overlapping
profile CPU observations are not added to wall time or compared across runs
as an isolated speed estimate. They motivate a separate, unlinked four-bin
half-stream prototype: retain the same FP16 FMA sequence and FP32 complex
reconstruction order while avoiding the full-frequency products buffer.
Require native bitwise equivalence and full approved-P PCM/hash regression
before any actual N1 measurement of that prototype.

## P half-stream pair and AR prefill attribution

Run `43177219205`, source `5a886e17e744996dbc38031c66333f5bb33ed180`,
completed on the same N1 2-vCPU / 8,000,000,000-byte / no-swap target. Three
excluded warmups and five measurements per case retained every slow sample.
The P baseline is an independent frozen-profile repeat; the candidate only
replaces full-frequency products/unpack with four-frequency scratch, preserving
the FP16 FMA sequence and FP32 complex reconstruction order.

| Runtime | Median RTF | Wall s | CPU s | AR s | Decoder s | Peak sampled tree RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P materialized products, independent repeat | 1.131582 | 5.680542 | 9.617176 | 2.044009 | 3.245276 | 6.370 |
| P-equivalent half-stream | 1.115283 | 5.598720 | 9.510815 | 1.919282 | 3.219002 | 6.306 |

All five candidate waveforms, tokens and 5.02s durations match N1 P exactly.
Model/checkpoint/reference/resource guards and the candidate's profile-to-own-
unprofiled PCM equality pass. Overall RTF decreases 1.44%, but decoder only
.81% and unchanged AR varies substantially again. This does not establish
sub-1 synthesis or a decisive decoder speedup. The raw pair and source/hash
provenance are in `results/n1_student_p_half_stream_20261112`.

The excluded profile now labels the actual `process_prompt` and decode paths
by matrix-storage identity, without retaining global operator input tensors.
Prefill is one `[1,256,512]` prompt over 18 blocks. QKV, output projection,
FFN1 and FFN2 each execute 18 FP32 addmm calls. FFN1 is `256x512 @ 512x2048`;
FFN2 is `256x2048 @ 2048x512`. Decode executes 118 steps, with 2,124 native
INT8 calls for each projection role, and classifier has its separate range.
No observed AR matrix is unattributed; acoustic addmm is outside these scopes.

Diagnostic prefill range totals are .1268s QKV, .1200s attention, .0484s output,
.1669s FFN1, .0063s activation and .1753s FFN2. Python dispatch labels add
overhead, so these totals are diagnostic attribution, not measured RTF stages
or values to add to wall time. The .3422s FFN pair motivates a separate FP32
packed-RHS/bias-ReLU prototype, not a claim that prefill alone closes the target.

Read-only cache analysis finds 2,150,572,032 bytes of original FP32 derived
spectral caches plus 1,075,286,016 bytes of additional FP16 derived caches.
These are not authoritative learned weights. A future opt-in lifecycle could
release unused FP32 derived caches while retaining parameters, but must rebuild
on mode changes/version mutations and needs separate correctness/memory/RTF
validation; no such release is implemented here.

## R: accepted pair bottlenecks, counterbalanced N1 comparison

Run `43180107755`, source `15bd6ae6a3c531472bb12d307fd708ec50f77cfb`,
uses the fixed N1 2-vCPU / 8,000,000,000-byte / no-swap profile. Five fresh
process pairs alternate L/R then R/L. Each process loads its own approved model,
records cold initialization, excludes three warmups, then measures one sample;
there are five measured observations per model. No P or new prefill arithmetic
is included. Every source/request/asset/parameter/resource guard passes.

| Runtime | Median RTF | Wall s | CPU s | AR s | Decoder s | Peak sampled tree RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L | 1.145699 | 5.751408 | 9.823378 | 1.915749 | 3.461919 | 5.107 |
| R, accepted K11 pairs48→36→48 | 1.141252 | 5.729087 | 9.728562 | 1.936707 | 3.353971 | 5.108 |

R has Korean listening approval, but the .39% overall RTF difference does not
establish meaningful complete-synthesis gain. Decoder decreases 3.12%; unchanged
AR varies in the other direction. Keep L as the operational baseline. Both
models retain the same 118 semantic tokens and 5.02s duration. The profile is
separate and exactly matches R's own unprofiled waveform. Raw cold/warm/slow
observations and source/hash provenance are in `results/n1_student_r_20261112`.

Actual profile tensors confirm `resblocks.8.convs1.1/.2` as `[36,48,11]`
on `[1,48,75520]` input and `convs2.1/.2` as `[48,36,11]` on
`[1,36,75520]` input. Their outer residual remains C48. Disjoint diagnostic
Tap scopes total about .710s for C48/C36 and .388s for C24. FFT streaming
records .776s self, with forward/inverse transforms .296/.286s and packing
.134s. Do not add these observations to measured wall time.

TorchDispatch labeling may produce both an outer dispatch and redispatched
operator entry in the profiler, so raw `operators[].calls` is not a kernel-call
count for instrumented AR. Read `ar_phases` for actual role/shape/invocation
counts. This diagnostic issue does not instrument the timed observations.

## Packed FP32 FFN prefill: complete N1 voice comparison

Run `43181086060`, source `80513b7c4ae7be0470c7d7cb535326034f9709db`,
measures only the explicit packed-RHS FP32 FFN1 bias/ReLU and FFN2 prefill
prototype against approved L. Decode stays W8A8/native attention and acoustics
stay FP32. The same fixed N1 2-vCPU / 8,000,000,000-byte / no-swap profile
uses five fresh-process pairs alternating L/packed and packed/L, each with
three excluded warmups and one measurement. R and P are not combined.

| Runtime | Median RTF | Wall s | CPU s | AR s | Decoder s | Peak sampled tree RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L | 1.160028 | 5.823341 | 9.901378 | 1.936976 | 3.481215 | 5.101 |
| Packed FP32 FFN prefill | 1.148206 | 5.763992 | 9.845126 | 1.977127 | 3.400939 | 5.338 |

All ten unprofiled waveforms, 118 semantic tokens and 5.02s lengths match
on N1; the separate candidate profile also matches its own unprofiled PCM.
All checkpoint/reference/parameter/resource guards pass. Overall RTF falls
1.02%, but changed AR grows 2.07%, while unchanged decoder falls 2.31%.
This does not demonstrate prefill speedup; retain L as operational baseline,
keep this prototype explicit and do not retune/repeat it without new evidence.
Raw cold/warm/slow observations and hashes are in
`results/n1_prefill_fp32_20261112`.

The excluded profile confirms 18 packed FFN1 and 18 packed FFN2 invocations
on the real `[1,256,512]` prefill. Diagnostic range totals are .154720s and
.156645s respectively, versus an earlier independent profile's original FFN1,
activation and FFN2 total of .348480s. This cross-run diagnostic difference is
not an isolated speed comparison and cannot override the measured AR result.
TorchDispatch produces outer and redispatched entries, so raw operator call
counts are doubled; the role metadata records actual invocations.

## L FFT geometry attribution for the next internal bottleneck

Run `43184533534`, source `0abf0c28847e92afe1b3620f004240ae7046fd07`,
is one unchanged approved L process on the actual fixed N1 resource profile.
It uses three warmups, one unprofiled PCM reference and one excluded profile.
No new model/arithmetic, packed prefill, R or P is combined. The profiled
waveform, semantic tokens and duration match its unprofiled reference exactly;
all asset/parameter/resource guards pass. This run makes no RTF improvement
claim. Source/raw hashes are in `results/n1_l_fft_geometry_20261112`.

| Actual FFT geometry | Modules | Input time | Diagnostic range total s |
| --- | ---: | ---: | ---: |
| C192 K7 | 6 | 18,880 | .412658 |
| C192 K11 | 6 | 18,880 | .429178 |
| C384 K7 | 6 | 2,360 | .197460 |
| C384 K11 | 6 | 2,360 | .207397 |
| C96 K11 | 6 | 37,760 | .337228 |

C192 is the largest grouped FFT scope (.841836s) in this diagnostic. Its K11
block is `resblocks.4`, original stage1/branch2, with convs1 dilations1/3/5
and convs2 dilation1. K7 is `resblocks.3`, stage1/original branch1. C192 K11
dilation1's four modules total .273422s, d3 .075466s, d5 .080290s; K7 totals
.269247/.069102/.074309s respectively. Geometry labels group identical
modules, so do not read the four-module d1 total as an individual convolution.

A candidate internal C192→144→192 pair preserves outer residual width and
native12 output alignment, but needs actual activation-based model selection,
recovery, listening and separate full-synthesis N1 comparison. The diagnostic
CPU ranges include profiler effects and must not be added to measured wall
time or treated as the speedup that a new model will deliver.

## U: approved C192 pair bottlenecks, isolated counterbalanced N1 comparison

Run `43189203151`, source `2915db55e6fc8baba7d87d1e9f11d3c109b979f4`,
compares unchanged approved L with owner-approved U on fixed N1 2-vCPU /
8,000,000,000-byte / no-swap resources. An isolated published branch contains
only approved U sources and the benchmark; unapproved T sources are absent.
HF revision `9287f847c7272ba2f45d0bd2a18512abec93a6e1` pins both bundles.
Five fresh-process rounds alternate L/U then U/L, each3warm1sample. The separate
U profile is excluded. No P, new prefill or combined candidate is measured.

| Runtime | Median RTF | Wall s | CPU s | AR s | Decoder s | Peak sampled tree RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L | 1.138691 | 5.716227 | 9.757876 | 2.021979 | 3.346141 | 5.100 |
| U, accepted additional C192→144→192 pairs | 1.137780 | 5.711656 | 9.668327 | 1.942282 | 3.392239 | 5.100 |

Both models retain identical118 semantic tokens and5.02s lengths. Every
checkpoint/parameter/reference/resource guard passes. U's profiled waveform
exactly matches its own unprofiled waveform and the preceding U process.
The overall median difference is only-.080%, while changed decoder grows1.38%
and unchanged AR varies in the other direction. Listening approval therefore
does not establish performance improvement. Keep L as operational baseline;
no retuned/repeated job is justified by this result alone. All24raw JSON
measurements and exact-source/HF/hash provenance are preserved in
`results/n1_student_u_20261112`.

Actual U profile shapes confirm selected `resblocks.4.convs1.0/.2` as
`[144,192,11]`, d1/d5, and `convs2.0/.2` as`[192,144,11]`, d1. The four
selected FFT scopes total approximately.2361s in this diagnostic; unchanged
K11 pair1 remains192→192. This verifies rectangular execution, not isolated
speedup: L's geometry profile was a separate run and scope labels add overhead.
Do not add analytical MAC savings or R/P/prefill differences to these timings.


## W genuine multirow INT8 prefill: recovered completed N1 result

Existing run [43195474472](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43195474472),
source `b88177079aff850861e5ab3848e190032e369b41`, completed successfully.
The handoff resumed by retrieving artifact13933846293, without dispatching
another measurement. Five fresh-process L/W observations alternate AB/BA;
each process excludes three warmups. The separate profile is excluded.

| Runtime | Median RTF | Wall s | CPU s | AR s | Decoder s | Peak sampled tree RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L | 1.145583 | 5.750826 | 9.802617 | 1.911403 | 3.459458 | 5.109 |
| W, genuine multirow W8A8 prompt FFN only | 1.124986 | 5.647429 | 9.567139 | 1.910003 | 3.357112 | 5.217 |

Every measured PCM/semantic hash and5.02s duration matches same-host L.
All original/student/checkpoint/parameter/resource guards pass. The overall
median is1.80% lower, but changed AR improves only.001399s(.073%), while
unchanged decoder improves.102346s(2.96%). This does not isolate a useful
prefill speedup. Retain L; do not combine W with V or claim its full1.80%
as an attributable gain.

The excluded W profile confirms18 genuine FFN1 and18 FFN2 calls; nested
TorchDispatch redispatch doubles low-level operator counts. Diagnostic
prefill totals are.478848s overall,.073471s FFN1 and.083318s FFN2.
These are instrumented scopes, not wall-time savings. Its PCM matches its
own unprofiled process exactly. Raw JSON, all slow observations, per-file
hashes and exact source/run provenance are in
`results/n1_student_w_prefill_20261112`; audio remains private locally.


## FP32 FFT output tiling4: direct SSH exact-output comparison

Source `462434b29253acaf1f1ee9e04081c6587f0a9623` runs on the same actual
Neoverse N1 2vCPU/8,000,000,000B/no-swap host through direct SSH, under the
single host benchmark mutex. It retains approved L/K weights, AR, transforms,
ACL8x12 inner-K accumulation and complex reconstruction order. Only output
channel groups are scheduled in chunks of4 in an isolated disposable DSO.
The default engine and native library files remain unchanged.

The synthetic primitive screening gives tile4/base ratios.9278 for C384,
.9836 for C192 and.9739 for C96. These are warm shape timings, not RTF.
Five fresh-process L/tile4 rounds alternate AB/BA,3warm1sample each, and
exclude the additional profile:

| Runtime | Median RTF | Wall s | CPU s | AR s | Decoder s | Peak sampled RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L | 1.149869 | 5.772341 | 9.851829 | 1.982425 | 3.410537 | 5.114 |
| Output tile4 | 1.143409 | 5.739916 | 9.812847 | 1.964784 | 3.385326 | 5.111 |

All10 measured PCM hashes,118 semantic tokens and5.02s lengths match L;
all resource/checkpoint/parameter/student guards pass. Every measured
candidate process uses the actual probe120times(3warm+1sample×30FFT).
The excluded profile also matches its own unprofiled PCM exactly.

The overall median change is-.562%; decoder changes-.739%, and unchanged
AR also changes-.890%. Decoder improves in only3of5 individual pairs.
This is insufficient to adopt the candidate or justify another same-setting
full measurement. Retain L; do not add the primitive gain or combine this
candidate with W/V/another runtime. Raw JSON, every observation, DSO/source
hashes and direct-SSH provenance are in `results/n1_fft_output_tile4_20261112`.
Generated audio and complete remote console logs remain private locally.


## Eight-row SDOT: exact compiler policy, hot versus rotating weights

Source `04bce68` screens an independent row8 operator on actual N1, keeping
scalar input quantization and per-output integer/dequantization order unchanged.
An earlier probe copied prefill's `-ffp-contract=off` policy and failed the
float-bit gate before any model inference. Matching the approved fast DSO's
complete flags/order/ABI restores exact values:40cases across5dimensions,
4finite scales and1/2threads pass int32-bit comparisons, immutable inputs,
and nonfinite-input rejection. The baseline DSO/weights are untouched.

| Matrix | Hot row8/base | Rotating64MB row8/base |
| --- | ---: | ---: |
| QKV1536×512 | .9939 | 1.0191 |
| Output512×512 | .9870 | .9995 |
| FFN1 2048×512 | .9718 | 1.0225 |
| FFN2 512×2048 | .9954 | .9026 |

The synthetic rotating pool better exposes weight traffic but remains a
primitive experiment, not full AR inference or RTF. QKV andFFN1 regress,
so the all-matrix row8 candidate is not adopted and does not justify a full
paired synthesis run. The FFN2-only observation is retained without combining
it with other changes. Raw values, manifests, initial failed-gate provenance
and exact source hashes are in `results/n1_int8_rows8_20261112`.


## Input quantization NEON only: bounded exact primitive screening

Source `d5f6437` keeps the original4row SDOT, scalar scale computation,
dequantization and complete approved fast compiler flags. Only finite amax
and input rounding/clamp/narrow use NEON. FRINTI follows current FPCR;
exceptional nonfinite reciprocal inputs fall back to the original scalar
expression. Baseline files and model tensors remain unchanged.

On actual fixed N1 resources,70bit-pattern conditions cover5dimensions,
1/2threads, zero/subnormal/ordinary/extreme finite scales from1e-40to1e37.
An identity-weight probe also checks signed zero and exact half-way ties.
Inputs are immutable and NaN/Inf is rejected.

| Matrix | Hot NEON/scalar | Rotating64MB NEON/scalar |
| --- | ---: | ---: |
| QKV1536×512 | .9813 | .9817 |
| Output512×512 | .9884 | .9930 |
| FFN1 2048×512 | .9669 | .9987 |
| FFN2 512×2048 | .8954 | .9528 |

Directions are consistent, but most weight-traffic dominated shapes improve
only slightly. This is neither model inference nor an RTF result, and does
not establish a useful full synthesis speedup. Keep the candidate isolated
and the default L path unchanged; no model/FFT/row8 candidate is combined.
Raw observations, exact flags/library/source hashes and finite-range scope
are in `results/n1_int8_quant_neon_20261112`.


## NEON quantization full-model PCM check, not speed adoption

Source `dc8af1654bf5d7fbe229fdecf5eafe6a6d473eb0` runs one fresh L process
and one fresh NEON-quantization-only process, each3excluded warmups+1sample,
on actual N1 2vCPU/8GB/no-swap. Runtime substitution is restricted to the
single disposable candidate process. Native call counter33984confirms
4inferences×118steps×18blocks×4linears reached the new quantizer.

Both processes retain identical118semantic tokens,5.02s duration and PCM
`0269c2ff38ef266552fb60743ff958a0376018280afa2197ef8bf01363069907`.
All original/student/checkpoint/parameter/resource guards pass; no audio
files are saved. The two single-sample RTF values1.139587/1.132694are
archival correctness-run timings, not evidence to adopt a speedup.
No additional paired measurement or new candidate is started.

Four actual baseline DSO/manifest SHA values captured during candidate
startup match the post-exit values. The measured wrapper's literal
immutability flag is not independent evidence; external raw snapshots and
loader hashes supply the check. Follow-up wrapper source `1a6bd18`
now computes its own actual before/after SHA comparison and fails on a
mismatch; running benchmark sources were not changed mid-process.
The initial standalone CLI lacked explicit prompt language and was rejected
before inference; this failed invocation remains in the raw evidence.
Raw JSON and full provenance are in
`results/n1_int8_quant_neon_20261112/full_pcm`. The host mutex is free and
no benchmark containers remain; pre-existing unrelated services are untouched.


## Approved AR18/V400: direct SSH counterbalanced N1 comparison

Source `73780c6820b0832ce0765ca80e14a780b17c0490` compares approved L with
owner-approved corrected AR18+original V400 on actual Neoverse N1,
2vCPU/8,000,000,000B/no-swap. The V manifest is
`def34d8555ea01c87091547fccbd27aa8a3bd33e8b6bbebb0df62095004c5d7c`;
its decoder is the unchanged original `7d4c184...b9524`, not V recovery100.
Both use identical approved K AR weights `4051df84...1335`. No new prefill,
FFT-size/tile, row8, NEON-quantization or FP16 candidate is enabled.

Five fresh-process pairs alternate L/V then V/L; every process excludes
three warmups and records one sample. The separate V profile is excluded.

| Model | Median RTF | Wall s | CPU s | AR s | Decoder s | Peak sampled RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L | 1.160668 | 5.826554 | 9.875800 | 1.912022 | 3.460650 | 5.120 |
| Approved AR18/V400 | 1.059591 | 5.319145 | 9.101680 | 1.958056 | 2.990778 | 4.886 |

All five paired decoder observations improve; their V/L ratios are
.93635/.87821/.86422/.88878/.82476. Decoder median decreases13.58%,
while unchanged AR median increases2.41%; the8.71% overall decrease therefore
has a credible changed-stage basis. RTF1 has not been reached. Every slow
sample remains in the record, including V RTF1.10142 and1.09475.

All ten processes use actual18AR blocks,118identical semantic tokens and
5.02s lengths. L PCM is `0269c2ff...69907`; V PCM `a2271807...4f1f5`
matches its approved listening sample exactly. All checkpoint, parameter,
student, reference and resource guards pass. The separate V profile matches
its own unprofiled waveform exactly.

Excluded diagnostic self times include spectral GEMM/reconstruction.5375s,
head attention.5142s, forward FFT.2441s, inverse FFT.2247s and packing.1453s.
These instrumented, potentially overlapping CPU scopes are not wall time or
isolated savings. Actual Cin240→Cout384 conv2 modules select FFT256;
`profile_geometry.json` records their8row padding and an unmeasured FFT128
alternative. No alternative numerical/quality/speed result is claimed.
All raw JSON, manifest/source/image hashes, per-pair ratios and shape costs
are in `results/n1_student_v_20261112`. Audio/console logs remain private.

### Approved AR17 QAT200 with unchanged V400, actual N1 (2026-11-12)

Source `c00e55f063d685e48ad56a661c265ee50d52d9cd`; owner-approved manifests
AR18+V `def34d8555ea01c87091547fccbd27aa8a3bd33e8b6bbebb0df62095004c5d7c`
and AR17+V `e30cdb7521a7e58053c554a2d78bf6f4d187d856b13083ff2d9e15987dc85bbb`.
The decoder is byte-identical original V400 in both. Runtime stays canonical
W8A8/native FP32 with normal FP32 acoustics; the pending head scheduling probe
is absent. Docker enforces 2vCPU / 8,000,000,000 bytes / no swap. Five fresh
processes per model alternate AB/BA, each with three excluded warmups and one
timed synthesis; both additional operator profiles are excluded.

| Median | AR18 + V400 | AR17 QAT200 + V400 |
|---|---:|---:|
| RTF | 1.057507 | 1.012739 |
| All five RTF range | 1.034626–1.101667 | 1.006747–1.023589 |
| Wall seconds | 5.308684 | 5.691592 |
| AR seconds | 1.928084 | 1.980327 |
| Decoder seconds | 2.942294 | 3.300265 |
| AR milliseconds / generated token | 16.339691 | 14.889673 |
| Decoder seconds / audio second | 0.586114 | 0.587236 |
| Generated tokens / audio seconds | 118 / 5.02 | 133 / 5.62 |

AR17's median RTF improves 4.23% and AR time/token improves 8.87%, while absolute
wall time increases 7.21% because its approved waveform is longer. This is a
model comparison with different generated sequences, not an isolated layer
compute speedup. Neither model reaches RTF1. Each of all ten observations
reproduces its own approved PCM/semantic hash, depth18/17, and duration; original
checkpoints, student assets, parameter and resource guards pass. Both separate
profiles reproduce their own model's PCM. AR17 is the next fixed realtime
baseline for the independent exact-output head scheduling trial. Complete raw
observations, guards, profiles, hashes and paired ratios are under
`results/n1_ar17_v400_20261112/`; no private model/audio files are committed.

### Exact head-value scheduling on approved AR17+V400 (2026-11-12)

Independent primitive source `c620b1a` passed all120 bit-pattern conditions,
including32dim production shapes,16/64dim fallback, ragged lengths/shared and
per-head/all-blocked masks, and1/2threads. For32dim weighted-value accumulation,
eight independent NEON vectors preserve every lane's original token/FMA order;
QK, softmax, masks and model tensors stay unchanged. The full comparison uses
source `75779d2a5eaf6cd82541e5610ced36eb47909319`, approved AR17+V400 manifest
`e30cdb7521a7e58053c554a2d78bf6f4d187d856b13083ff2d9e15987dc85bbb`,
canonical old DSOs and an explicitly registered standalone probe. Five fresh
processes/runtime alternate AB/BA on the same2vCPU/8e9byte/no-swap N1, each
with3excluded warmups+1sample. One separate candidate profile is excluded.

| Median | Unchanged runtime | Value schedule |
|---|---:|---:|
| RTF | 1.016054 | 0.992652 |
| All five RTF range | 1.004976–1.042967 | 0.990021–1.008962 |
| Wall seconds | 5.710221 | 5.578705 |
| AR seconds | 2.126797 | 1.896119 |
| Decoder seconds | 3.291303 | 3.289368 |

All5paired observations improve total RTF and AR time. The median paired
candidate/baseline ratios are0.977140 overall and0.942323 AR; unchanged decoder
ratios range0.989462–1.011574. Independent stage medians must not be added or
interpreted as the paired causal effect. The candidate median falls below1,
and4/5candidate observations are below1; the worst is1.008962. This is evidence
for this fixed Korean request/model/resource profile, not a guarantee that every
request/run is realtime. All11processes reproduce approved133tokens/5.62s PCM
and semantic hashes with original/student/parameter/resource guards intact.
Every candidate timed process records9044actual kernel calls
(4inferences×17layers×133tokens) and actual baseline4file SHA before/after
matches. The excluded profile records11305kernel calls and its own PCM exact.

Canonical adoption is `e9e6442`: the same32dim loop moves to the normal native
head function, without the probe's atomic counter;16/64dim behavior stays on
the original loop. Head geometry and the source-hashed build manifest identify
the selected production path. Source `c356245` adds an independent five-fresh
canonical confirmation plus an excluded operator-profile invocation check;
that actual run is pending at this evidence snapshot. Primitive and all full
raw observations/guards/profiles/provenance are in
`results/n1_head_v32_20261112/`.

Canonical confirmation source `c35624599dac169ad752819383f515ff441b0aeb`
finished with fresh RTFs1.011936,1.010217,1.003143,1.017123,0.991603: median
**1.010217**, range0.991603–1.017123. Wall median5.677420s, AR1.897647s and
decoder3.288318s. The canonical operator executes in the excluded profile
(`head_attn`4522reported events); all17blocks report16heads×32dims and the
source-verified rebuilt DSO declares the adopted schedule. No research
substitution/counter is loaded. All6processes reproduce the approved PCM and
133tokens/5.62s with unchanged original/student/parameters/native files and
resource guards. Full raw independent evidence is under
`results/n1_head_v32_20261112/integrated/`.

The integrated default path therefore **has not established stable RTF<1**.
Do not pool this independent run with the controlled prototype comparison or
add samples merely to pass1. The exact scheduling change stays adopted based
on all5controlled paired AR/RTF improvements; next work is an independent QK
scheduling hypothesis on the same approved AR17 model. Existing Actions choices
now expose `studentardepth`, `studentheadv32` and `studentheadconfirm`; each
requires3warm5samples. Reproduce the historical pre-adoption causal head
comparison from source75779d2 (latest canonical already includes its value
loop), and use `studentheadconfirm` for current canonical path verification.
Private model bundles must be available in the caller's pinned approved private
asset revision/cache; no model upload or workflow dispatch was performed here.

### Isolated four-token QK scheduling primitive (2026-11-12)

Source `486b2cbc591a4e4163ea70dfe12f60f217005837` compares the adopted canonical
value schedule against independent QK scheduling. The32dim path loads the eight
query vectors once/head and interleaves four independent token accumulators;
each token still performs its original dimension-order FMAs and horizontal
reduction, and scores/max update in increasing token order. Canonical value
scheduling, softmax and masks stay unchanged. Other dimensions and ragged token
tails retain the original QK loop. All120bit-pattern conditions pass with
immutable inputs; compiled flags exactly match canonical and four baseline
DSO/manifest file hashes match before/after. Actual Docker profile is
2vCPU/8e9bytes/no swap, resource guard exit0.

For lengths257/315/374/768, hot function ratios are0.5348/0.5526/0.5357/0.5105;
17-layer rotating-KV ratios are0.6275/0.7130/0.6018/0.7016. These are synthetic
function timings, not full synthesis or RTF predictions; actual AR also streams
weights. Raw timing arrays,120conditions, native/build hashes and resource
report are in `results/n1_head_qk4_20261112/`. Full PCM-gated fixed-model AR17+
originalV400 comparison source `5cb102c` is pending. Rejected AR16 is excluded
from every performance run.

Full QK4 source `5cb102c7ca4bc49c737e0cbde3d47fadc953c6d2` finished all5pairs
and an excluded profile. Baseline RTFs1.029649,.990742,.996747,.999784,.991685;
QK4 .989601,.981041,.998626,.996269,1.019419. Medians .996747→.996269 differ
only−0.048%, with overlapping ranges .990742–1.029649/.981041–1.019419. AR
median worsens1.865049→1.918634s; CPU medians9.580549→9.555400s differ−.26%.
Only3/5pairs improve overallRTF/AR. All11processes pass approved133tokens5.62s
PCM/semantic, original/student/parameter/resource guards. Candidate processes
record9044nativecalls and actual baseline4file SHA matches; profile11305calls
and ownPCM exact. Raw all24JSON and ratio arrays are in
`results/n1_head_qk4_20261112/full_pcm/`.

**QK4 is not adopted.** The strong primitive does not establish a full-runtime
win; no canonical QK modification or extra confirmation/sample collection is
performed. Keep the approved AR17+originalV400 model and canonical exact value
loop. Next isolated hypothesis is row8 only forFFN2 shape512×2048, preserving
row4 everywhere else; do not mix QK4. Its earlier64MBrotating ratio .9026 is
promising for thatshape, but FFN2 accounts for roughly one-third of four-GEMV
MACs, so only a small aggregate gain is plausible and must be measured.

### FFN2-only row8 and excluded addmm attribution (2026-11-12)

FFN2 sourcee269bd5 routes only512×2048 to row8, all other shapes to original
row4; approved AR17+V400 and canonical value scheduling stay fixed, QK4 absent.
All70bit cases (4/12rowfallback, realshapes, finite magnitudes/signed0/1–2threads)
and68approved-model weight matrices/syntheticactivations match exactly. Actual
17layer/53,477,376byte mixed4GEMV traversal ratio .971738, FFN2-only .922705,
hot .985455. Correctness source0bdc1ee completes3warm1sample with approved
133tokens5.62s PCM exact,9044FFN2row8+27132row4fallbackcalls, baseline4file
SHA/model/parameter/resource guards intact. Its singleRTF1.002543 is not speed
or worst-case evidence. No default adoption or5pairedclaim; prioritize larger
acoustic bottlenecks. Raw arrays/proof: `results/n1_ffn2_rows8_20261112/`.

Opt-in `--profile-matrix-shapes` source2be5404 attributes one excluded profile
without retaining input tensors globally. All284actual mm/addmm calls are mapped:
ARprefill68, Tapmm24 and Tapaddmm.out192; predictor uses native lossless_linear,
not addmm. Instrumented processCPU/wall bygroup: ARprefill .7766/.4108s;
C48K7 .5296/.3767; C48K11(rect36) .6455/.3361; C24K7 .2780/.1453;
C24K11 .4345/.2273. C48K11 accounts for34.2% ofTapCPU, lastC24K11 for23.0%.
These include profiling overhead and must not be converted into production wall
savings. Own-profile PCM exactly matches approved133tokens5.62s, all guards pass.
Raw per-module/tap matrix identity/shape/cost and source hashes are in
`results/n1_addmm_attribution_20261112/`. FFT128-on-Cin240 is the next separate
fixed-weights quality/numeric candidate; no approval or performance claim yet.
