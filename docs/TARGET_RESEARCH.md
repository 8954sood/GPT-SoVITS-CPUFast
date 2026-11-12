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
