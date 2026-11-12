# W prefill runtime: operation and restore notes

This file covers W only. The broader environment pause, model inventory and
student work are documented separately in `ENVIRONMENT_HANDOFF.md`.

W keeps approved L/K model tensors, W8A8 decode/native FP32 attention and normal
FP32 acoustics. Only prompt FFN1/FFN2 use the isolated genuine multirow W8A8
GEMM. Its 4-token×4-output SDOT tiles share RHS loads across tokens; this is not
repeated single-token GEMV. FFN1 bias/ReLU is fused into final stores. Original
parameters remain FP32 and unchanged; derived INT8 buffers are disposable.

Korean listening scope is inherited only because local W and approved L have
identical PCM, 118 semantic tokens and 5.02s duration for the fixed reference,
text and seed 42. Japanese/English differ and are not approved. This records
no additional listening answer from the owner. The source/equality proof is
`results/student_pilot/approved_w_runtime.json`; its seven source hashes and
baseline approval/report hashes are checked before the benchmark loads TTS.

## Restore the published source

Use repository `8954sood/GPT-SoVITS-CPUFast`, branch `codex/cpu-w-prefill`.
For an exact reproduction, select `head_sha` from the completed W measurement's
`provenance.json`, rather than assuming a branch tip remains unchanged. This
isolated branch starts at published U result commit 422e191 and adds only W
primitive/helper/proof, benchmark/workflow and operating notes. T code and new
unapproved V weights are not included. All model files remain outside Git.

`--int8-prefill-ffn` is opt-in. The default L engine is unchanged. It requires
`--ar-int8-only`, Korean and the fixed approved K bundle; it cannot be combined
with packed FP32 prefill or FP16 acoustic candidates. On N1, candidate PCM,
tokens and duration must exactly match the same-host L comparison or the job
fails at the listening gate. Every source/checkpoint/parameter/resource guard
must also pass. A separate W profile must preserve its own unprofiled PCM.

## Runtime and native builds

The measured host is Linux ARM64 Neoverse N1 with 2 vCPU,
8,000,000,000 bytes RAM and no swap. Local Mac/Docker results are correctness
checks, not substitute performance measurements. Existing benchmark Docker
images and original model assets are separately supplied; this repository does
not distribute the image or those assets. The current remote script defaults
to image `cpufast-n1:cefae22`, overridden by `ARM_BENCH_IMAGE`.

Build within the actual Linux ARM64 Python/PyTorch container. A C++17 compiler,
OpenMP and ARM dot-product support are required. The W builder uses
`-march=armv8.2-a+dotprod`, `-mtune=neoverse-n1`, `-ffp-contract=off` and
`-fno-fast-math`. Rebuild after changing architecture, PyTorch, ABI or sources;
DSO manifests bind source hashes, library hash and PyTorch version. Do not
restore a Mac-built or differently versioned binary into the measured runtime.

```bash
python build_cpufast_ar.py --output libcpufast_ar_torch.so
python build_cpufast_fast.py --output libcpufast_fast.so
python experiments/rtf07/build_prefill_int8_probe.py --output libcpufast_prefill_int8_probe.so
export CPUFAST_AR_LIBRARY="$PWD/libcpufast_ar_torch.so"
export CPUFAST_FAST_LIBRARY="$PWD/libcpufast_fast.so"
export CPUFAST_INT8_PREFILL_LIBRARY="$PWD/libcpufast_prefill_int8_probe.so"
python experiments/rtf07/test_prefill_int8_probe.py
```

The primitive's tests compare to the explicitly defined per-token W8A8
reference, not original FP32 output. Tests cover the real FFN dimensions,
ragged tiles, zero values, original K validation, 1/2 threads, immutable inputs,
version invalidation and current bias. The prompt proxy restores MLP methods
on success/error and delegates decode to the original runtime. Replacing
prefill arithmetic can still change later KV values and speech, despite leaving
attention/cache-management code unchanged.

## Private assets and GitHub Actions

Always pass `-R 8954sood/GPT-SoVITS-CPUFast` to `gh`: the fork's inferred
repository may otherwise point upstream. W uses one immutable existing K
bundle, not new weights:

- HF repository: `babihoba/gpt-sovits-v2proplus-arm-students` (private).
- Revision: `475ac97f4a503ea9c6581333c8b936fd5c9f6dcd`.
- Manifest: `approved-k-20261112/manifest.json`.
- Manifest SHA256: `761a3dd80e40211d469823a2fffd8c3c5d153162790029bae12cf152cc7a17b4`.
- AR SHA256: `4051df84ee71e52eeb9858a53a060c4785eef912cf25a51051069ddd99041335`.
- Decoder SHA256: `a2fe6478aae00a8a481209f32aba85eab97198cd30aca0ce1ff63d9ec31c3c1d`.

Repository secrets required by `.github/workflows/arm-rtf.yml` are
`ARM_SSH_HOST`, `ARM_SSH_USER`, `ARM_SSH_PRIVATE_KEY`,
`ARM_SSH_PASSPHRASE`, `ARM_SSH_KNOWN_HOSTS`, and `HF_STUDENT_TOKEN`.
Record their names, never values. Optional repository variables are
`ARM_BENCH_ROOT`, `ARM_BENCH_IMAGE`, `ARM_BENCH_MEMORY` and `ARM_BENCH_CPUS`.
The performance guard requires 2 CPUs,8,000,000,000 bytes and swap 0 regardless of
where the workflow is launched. SSH requires the configured known-host key and
an account allowed to run Docker (directly or via noninteractive sudo).

The remote asset layout is defined at the top of
`scripts/run_arm_benchmark.sh`: `$ARM_BENCH_ROOT/project` supplies original AR,
full HuBERT and Japanese resources; `cpufast-cefae22/assets/pretrained_models`
supplies original S2/speaker assets; `reference_ja_4s.wav` is the fixed private
reference. Private approved bundle files are verified and reused under
`$ARM_BENCH_ROOT/student-assets/<manifestSHA>` and mounted read-only. Actions
uses only a temporary downloader/token home and disposable source checkout;
cleanup removes temporary upload/checkout data. Weights and the input reference
are not public artifacts. Generated audio and numerical JSON are artifacts.

This is the exact W dispatch shape. Record and check the returned run's
`headSha` against the reviewed source before treating it as the intended run:

```bash
gh workflow run arm-rtf.yml -R 8954sood/GPT-SoVITS-CPUFast \
  --ref codex/cpu-w-prefill \
  -f mode=studentprefillint8 -f warmups=3 -f samples=5 \
  -f student_repo=babihoba/gpt-sovits-v2proplus-arm-students \
  -f student_revision=475ac97f4a503ea9c6581333c8b936fd5c9f6dcd \
  -f student_manifest_path=approved-k-20261112/manifest.json \
  -f student_manifest_sha256=761a3dd80e40211d469823a2fffd8c3c5d153162790029bae12cf152cc7a17b4
```

Mode `studentprefillint8` builds both existing native libraries plus the separate
W DSO, runs the native preflight, then executes five fresh-process alternating
L/W and W/L rounds. Each process has 3 excluded warmups and 1 measurement. A separate
W phase profile is excluded from the five observations. Keep every cold/warm/
slow sample, CPU seconds, peak RSS, asset/reference SHA and semantic/audio
checks. Read `paired_aggregate` in `student_summary.json`; do not pick one fast
sample or add differences from independent R/P/U/prefill runs. Profiler CPU
scopes overlap and carry instrumentation overhead, so they are not wall time.

Workflow concurrency must remain `cpufast-arm-benchmark`, with
`cancel-in-progress: false`. Only one heavy measurement may use the server at a
time. During local training/listening, coordinate the single Docker VM slot;
use local runs only for correctness and listening. No new remote queue belongs
to this handoff after the owner-requested current W test; resume experiments
only under the next environment's explicit task.

## Files to keep, scratch and handoff boundary

Keep source, native builder/manifests, the W equality proof and final raw JSON
plus exact run/source provenance. DSO files are small, rebuildable environment
assets. W has no new model checkpoint; reuse the verified K bundle. Native unit
tests require no model/context files. Large block captures were only temporary
correctness data and are unnecessary to run W. Never duplicate training corpus
or checkpoints merely to give another agent access: use explicit read-only
paths/mounts or the approved immutable private HF release.

For the next environment, split responsibilities as in this task: the model
owner handles student/spec/train/listening assets, and one CPU worker owns
native/engine/benchmark/workflow. Pin source/HF/input/seed, coordinate compute
slots, keep new candidates opt-in, obtain quality evidence, then measure the
actual server. Do not spawn extra agents by default or let separate agents
launch overlapping server jobs. Existing failure records in
`docs/TARGET_RESEARCH.md` rule out blind retries of tap tiling, im2col, packed
FP32 prefill and small noisy R/P/U differences.
