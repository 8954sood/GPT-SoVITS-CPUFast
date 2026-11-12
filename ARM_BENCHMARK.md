# ARM64 experimental RTF benchmark

The manual **ARM RTF benchmark** workflow builds the FP32 native runtime on the target host, runs native and tap-kernel checks, then measures a baseline and the small-channel tap candidate on two Japanese sentences in serial. Each uses the same reference, seed and model checkpoints. JSON artifacts contain individual warmup and measured runs, guard results, CPU quota, timings and parameter audits. Generated WAV files for the four runs are also uploaded for listening; the input reference and model files are never uploaded. Model loading is excluded from RTF, and output includes 0.3 seconds of appended silence.

The FP32 engine uses AR prefetch distance 256 by default, based on two actual-server comparisons with PCM-exact outputs; pass `--ar-prefetch 0` to disable it. This workflow uses the FP32 path with value-preserving AR weight caches; it does not select the separate experimental W8A8/FP16 path. Checkpoints and authoritative AR/acoustic parameter hashes are checked before and after each run. Numerical rounding can differ and the generated audio still needs listening review. The default is the configured **actual 2-vCPU ARM server**, not the local 0.7426-CPU proxy; never compare their raw RTFs as if they were the same environment. The workflow is manually dispatched only.

## Configure a target host

The workflow runs on a GitHub-hosted runner and connects over SSH to the ARM server; the runner itself does not need to be ARM. It requires a prepared, trusted host with Docker (direct access or passwordless `sudo -n docker`), at least 8 GB available to Docker, and the following private assets. Do not commit the SSH key, model assets, or reference audio.

```
$HOME/gpt-sovits-benchmark/
  reference_ja_4s.wav
  project/models/s1v3.ckpt
  project/models/chinese-hubert-base/
  project/.venv/lib/python3.12/site-packages/pyopenjtalk/open_jtalk_dic_utf_8-1.11/
  cpufast-cefae22/assets/pretrained_models/v2Pro/s2Gv2ProPlus.pth
  cpufast-cefae22/assets/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt
  cpufast-cefae22/assets/pretrained_models/fast_langdetect/lid.176.bin
```

The default image `cpufast-n1:cefae22` is a **locally prepared image**, not published by this repository. An alternate host needs an equivalent Linux ARM64 image with Python 3.12, PyTorch 2.8.0, the project inference dependencies, OpenMP, and a C++ compiler capable of building PyTorch extensions. The image must run `python` by default and make the dependencies available to a non-root user. The mounted Open JTalk dictionary and language identification cache must be readable. Model weights and the reference must match to compare results. This workflow does not download or provision models or the image.

In repository **Settings → Secrets and variables → Actions**, configure these repository secrets for the target: `ARM_SSH_HOST`, `ARM_SSH_USER`, `ARM_SSH_PRIVATE_KEY`, `ARM_SSH_PASSPHRASE` (empty if unencrypted), and `ARM_SSH_KNOWN_HOSTS` (a verified known_hosts entry). Restrict repository write access: anyone allowed to change and run the workflow can use its secrets. To change hosts, update all relevant secrets, especially the verified host key.

Optional repository variables: `ARM_BENCH_ROOT` (absolute path to the layout above), `ARM_BENCH_IMAGE` (local image name), `ARM_BENCH_CPUS` (Docker CPU quota, default `2`), and `ARM_BENCH_MEMORY` (bytes, default `8000000000`). The experiment uses two inference threads even if a different CPU quota is selected; record that distinction when comparing hosts. Docker mounts the asset directory read-write for runtime caches, with the S2 and speaker-verification checkpoints individually mounted read-only; the project and reference are also mounted read-only.

## Run

Select `compare` for the two-sentence audio comparison , `diagnostics` for a separate operator profile and synthetic FP32 kernel probes, `attention` to compare the approved tap path with short-context native attention on two sentences, `small` for synthetic small-channel convolution probes, `scheduling` for an active/passive OpenMP wait comparison, `packed48` to compare the selected 48-channel convolution candidate against tap on two sentences, `fft` for synthetic FFT-size tuning, `fftcandidate` for an end-to-end FFT-size comparison and AR cache probe, `arcache` for full synthesis with FP32 AR coefficient caches, `prefetch` for model-sized AR cache traversal, `prefetchcandidate` for the selected full-synthesis prefetch comparison, `winograd` to probe three-tap decoder methods, `direct3` to compare a selected direct-convolution candidate in full synthesis, or `locality` to probe AR prefetch locality. Profiling runs after the measured synthesis and is excluded from RTF; synthetic timings are not end-to-end performance results.

Open **Actions → ARM RTF benchmark → Run workflow**, choose the trusted branch and warmup/sample counts. The summary reports median and individual RTF; download the JSON and WAV artifacts from the run for details and listening. A temporary checkout is removed from the server after the run. The default workload uses the Japanese reference and text specified in `scripts/run_arm_benchmark.sh`, with seed 42. Hardware load, CPU quota, memory, runtime image and thermal conditions can affect the result.

To run directly on an already prepared target from a checkout:

```sh
bash scripts/run_arm_benchmark.sh "$HOME/gpt-sovits-benchmark" cpufast-n1:cefae22 8000000000 2 2 3
```

The outputs are `ci-results/{baseline,tap}_{1,2}.json`, their guard reports and corresponding WAV files. The script rebuilds the FP32 native extension in that checkout.

`mode=fftstream` compares the independently verified FP32 tap/prefetch-256
baseline against an opt-in four-frequency streaming spectral GEMM. It retains
the same ACL FP32 kernel and complex reconstruction order, but writes the
complex inverse-FFT layout from bounded scratch storage rather than retaining
all three frequency-product planes. The job first checks bit-exact synthetic
products and convolution epilogues, then reports a synthetic probe and full
synthesis of both fixed sentences. This is a candidate, with no N1 performance
claim until the action completes; the default engine remains unchanged.

`mode=transforms` separately tests channel-parallel Winograd input/output and
FFT finishing (`--parallel-transforms`). Independent channels retain the same
FP32 arithmetic and output order; this is not branch-parallel synthesis.
The default remains serial transforms until target full-TTS repeats justify a
change. Native tests compare serial/parallel outputs exactly, including
non-contiguous residual channel rows.

`mode=inference` compares the same original FP32 profile under its existing
`no_grad` path and the benchmark's explicit `torch.inference_mode()` context.
Model loading and installation remain outside that context, preserving normal
parameter version counters and the cache-validity checks. Both variants use
fresh processes, with every warmup/sample recorded. Local checks establish
correctness only; any performance claim must use the target action.

`mode=runtime` batches an independent original-FP32 baseline, parallel transforms,
inference mode, and their combination with FFT streaming on both fixed
sentences. The non-baseline order is reversed for the second sentence. Use
three excluded warmups and five retained samples for confirmation. Every
candidate is compared to that sentence's baseline WAV, and the combined RTF is
measured directly. Individual improvements are not added together.

For a separately approved model requiring FP32 AR storage, the experimental
`--ar-fp32-cache --ar-prefetch 256` combination is now supported. Unspecified
prefetch with FP32 cache defaults to zero. This option passed local ARM64
bit-exact operator checks only and has no target-server speed claim; do not
infer a benefit from the original model's exact-half-cache prefetch results.

## Verified original-model runtime profile

Actual-server five-sample `mode=runtime` verification measured the combined
original FP32 configuration at median RTF **1.4732 / 1.4464**, compared with
**1.5340 / 1.5501** for its baseline on the two fixed Japanese sentences.
All candidate PCM outputs were exact. The target RTF below 1 remains open.
Raw reports/provenance are in `results/n1_runtime_20261112`; see
`docs/TARGET_RESEARCH.md` for the independent comparisons.

Build the current native library first. For an already initialized, evaluation
FP32 V2ProPlus CPU pipeline, keep model loading and installation outside
inference mode so original parameter version checks remain available:

```python
import torch
from cpufast_engine import optimize

engine = optimize(
    pipeline,
    head_major=True,
    cook_phase=True,
    fft_layout=True,
    packed_fft=True,
    fused_finish=True,
    tap_small=True,
    fft_stream=True,
    parallel_transforms=True,
    ar_prefetch=256,
    ar_threads=2,
    acoustic_threads=2,
)
with torch.inference_mode():
    chunks = list(engine.run(request))
```

The benchmark equivalent adds `--fft-stream --parallel-transforms
--inference-mode` to the existing approved tap/prefetch profile. These options
remain explicit and preserve original-model coefficients. The combined score
is measured separately for approved G below. A fresh pipeline is required when
changing weights/settings; rebuild the extension after native source changes.

## Approved private student benchmark

`mode=student` fetches only the approved manifest and its three model/spec files
from a private Hugging Face model repository at a full immutable commit SHA.
Supply `student_repo`, `student_revision`, `student_manifest_path`, and the
independently verified `student_manifest_sha256`. The runner uses the repository
secret `HF_STUDENT_TOKEN` without login or persisted credentials. Model bytes
are never added to Git or public Actions artifacts. Download errors report only
type/status because signed storage URLs may include temporary credentials.

Before transfer, after private SSH installation, and before/after inference,
all asset sizes/hashes are checked. The server keeps a user-private cache under
`ARM_BENCH_ROOT/student-assets/<manifest SHA256>` and mounts it read-only.
Temporary runner copies/cache and the disposable source/upload directory are
removed; the verified server cache is reusable. Reference audio stays on the
server and is checked against the approved manifest.

The serial matrix measures Korean original combined-runtime baseline, then
approved G with FP32-cache prefetch 0/64/128/256 under fixed decoder settings.
It keeps 0 unless the best hint improves the measured median by at least 1%;
the selected hint remains provisional. A fresh combined-kernel G process has
five retained measurements. Japanese/English measure original/G combined only;
these languages have synthesis checks and are not separately listening approved.
The additional operator profile runs last, excludes shape retention to bound
memory, and is excluded from the main RTF table. Every warmup/slow measurement,
CPU time, stage, RSS, semantic token count/hash and waveform check is retained.
Original/Student durations may differ; compare wall/CPU times together with RTF.

A student cannot be loaded unless the trusted approved manifest, checkpoint
SHA/size, original S2/reference hash, architecture, text, prompt language and
seed match. Original checkpoint files remain unchanged. Within-student prefetch
and combined candidates must produce the same PCM/tokens/duration as N1 G0.
Local approved-waveform matches are reported separately from the N1 comparisons.

`mode=studentpredictor` is a separate, opt-in follow-up for the same immutable
approved G bundle. Supply `student_prefetch` from the preceding N1 sweep. It
compares the combined G baseline against `--predictor-eligibility-cache` with
five retained measurements each on Korean, then records a separate candidate
operator profile. The optimization caches only a failed FP16-storage eligibility
check; the predictor continues to execute its original authoritative FP32
`F.linear`. Parameter ID/version/storage/dtype/device changes invalidate the
negative cache. This mode must pass bit-exact PCM/token/duration checks before
any adoption; preparing its code is not a performance claim.

The first approved G actual-N1 sweep completed in
[run 43161518437](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43161518437).
Korean FP32-cache prefetch 0/64/128/256 medians were
**1.4709 / 1.4779 / 1.4786 / 1.5155**; retain **0**. The independent five-sample
G combined result was **1.4414**, while the three-sample original combined
baseline was **1.4440**. Original/G wall times were 8.346/8.504s for different
output lengths, 5.78/5.90s. Layer reduction therefore did not deliver a meaningful
Korean synthesis speed improvement on this server. Trained FP32 AR storage
increased AR time while the decoder became faster. Full data, every slow sample,
guards, and the separate profile are in `results/n1_student_g_20261112`.
Within-N1 Korean variants were PCM/token exact. All three generated semantic
hashes matched approved local synthesis; cross-platform PCM hashes differed.
Japanese/English are performance smoke checks only, outside listening approval.

`mode=studentcompare` compares two separately listening-approved bundles in one
serial N1 job: baseline and candidate each have three warmups and five Korean
measurements, followed by a separate candidate profile. For an exact-half
candidate, a third case runs those same approved coefficients in a full FP32
cache with `--student-fp32-control`; both storage paths must produce identical
PCM/tokens/duration. This separates the effect of cache storage/runtime from
the separately approved coefficient rounding. Supply the candidate's
usual manifest inputs plus `comparison_manifest_path` and
`comparison_manifest_sha256` for the baseline in the same immutable private HF
revision. Both verified bundles are mounted read-only. There is no cross-model
PCM equality requirement because model changes can alter tokens and duration;
each report retains its own approved semantic/PCM matches and asset hashes.
The extra profile is checked against the candidate's own timed waveform.

Manifests may declare `model_variant` and `ar_cache_storage`, either `fp32`
(the backward-compatible G default) or `exact_fp16`. The latter is a separately
approved inference-copy checkpoint whose AR matrix values must pass the existing
exact half round-trip guard. Arithmetic remains FP32; no benchmark code rounds
weights. The student spec's original layer IDs must be unique, within 0–23,
and match the manifest/loaded AR layer count. Decoder branch identities and
original teacher/reference hashes also remain checked. FP32 cache uses the
explicit `student_prefetch` from its prior N1 sweep; exact-half cache uses the
verified original profile's distance 256. Each row records the actual setting.
`studentcomparecached` additionally enables the same opt-in predictor negative
eligibility cache for both models; use only after its independent N1 comparison.

The independent predictor negative-cache comparison
[run 43163928379](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43163928379)
did **not** improve G: baseline/candidate medians were **1.4002/1.4047** over
five samples each, despite a small AR/CPU reduction. All candidate waveforms and
tokens were exact. This is a failed speed hypothesis; keep the option off.
Raw reports/provenance are in `results/n1_student_predictor_20261112`.
The classifier range and its input/weight shapes are now recorded only during
the additional excluded profile synthesis, with no wrappers in timed samples.
