# Direct SSH continuation, 2026-11-12

The owner explicitly resumed the transferred task with one CPU performance
worker using direct SSH. This supersedes the previous chat's pause instruction.
The transfer was restored in a fresh checkout from `a854031`; existing projects
and original checkpoints were preserved.

## Environment and ownership

- All release attachment sizes and SHA256 values in `TRANSFER_RECEIPTS.json`
  were checked. The research archive's 206 files and original archive's 29 files
  also passed their internal SHA256 indexes. Tar hard links were resolved to
  verified files; no untrusted links were followed into external directories.
- Python 3.12 / Torch and torchaudio 2.8.0 CPU training imports passed. The
  student assets, matrix, geometry, runtime, loss and QAT host checks passed
  28 tests with 2 ARM-native skips. Private restore evidence and the installed
  package versions are under `.experiments/handoff/`.
- CPU performance work runs in a separate worktree. One worker owns the server
  queue. `scripts/run_arm_benchmark.sh` uses a host file lock to serialize direct
  SSH and Actions runs that use this script.
- The host remains Linux ARM64 Neoverse N1. Performance measurements require
  2 vCPUs, 8,000,000,000 bytes and no swap, irrespective of physical host RAM.
  Local CPU training and quality synthesis timings are not N1 RTF evidence.
- SSH key material remains outside Git, in a task-specific agent. Private model,
  corpus, reference and generated audio files remain outside tracked sources.

## Recovered W result

Existing Actions run `43195474472` completed successfully at source
`b88177079aff850861e5ab3848e190032e369b41`. Its artifact was recovered without
dispatching the experiment again. Raw numerical results and provenance are in
`results/n1_student_w_prefill_20261112/`.

L/W median RTF was 1.145583 / 1.124986. AR time was essentially unchanged:
1.911403 / 1.910003 seconds. Most of the wall-time difference occurred in the
unchanged decoder. This is insufficient evidence to adopt W as a prefill
speedup; L remains the performance baseline. See `TARGET_RESEARCH.md`.

## V listening provenance correction

The inherited description calls V “approved L AR plus V decoder.” The archived
`listening_V/comparison.json` actually records 24 AR layers, native audit blocks
0–23 and 137 Korean semantic tokens / 5.78 seconds. The archived L and W
reports record 18 layers and 118 tokens / 5.02 seconds. Consequently the old
V listening feedback cannot establish approval of the different AR18+V
combination. Preserve the historical report rather than rewriting its claims
or exporting a falsely approved AR18 bundle.

`synthesize_student.py` now checks actual transformer blocks against `ar_keep`
for student B/D synthesis and records input hashes and actual block count.
Prepare fresh same-host original A, L18 and V18 listening files first. Only
after the owner accepts the correct combination should an immutable V
performance bundle be exported and a separate L/V comparison run.

The authoritative V checkpoint SHA256 remains
`7d4c184c638b09880a7e61b088b999542598644d22166e84e32137c5649b9524`;
the spec remains
`cb411065f3e69c2725791fc00272940589c309a67bd5122abcd26499cfde6a16`.
Its 159 state tensors contain 22,811,196 parameters. Restored CPU evaluation
reproduced every inherited grouped validation log-mel value exactly. On the
same 12 validation contexts, mean original-teacher log-mel error is 0.117331
for L and 0.162211 for V. These are numerical decoder errors, not listening
ratings or free-generation quality guarantees.

A separate CPU-only 100-update recovery candidate completed from the
authoritative V checkpoint at learning rate 2e-6 with cosine decay, accumulation
3, full contexts and original-teacher log-mel supervision. It uses a fresh
private output directory `budget40_recovery_20261112`, retaining the original
checkpoint. Its fixed validation mean log-mel error decreased from 0.162211
to 0.160459, with reductions in all six language/latent-origin groups. This is
a small numerical improvement, not a listening or speed approval. Details are
in `results/student_pilot/v_recovery100_cpu_20261112.json`. The redundant latest
checkpoint was removed only after confirming its step/spec/tensors exactly
matched the retained best checkpoint.

## Bounded native runtime experiments

One worker ran three independent hypotheses serially over direct SSH. The
default engine, FFT implementation, AR implementation and baseline native
builders were unchanged. No candidate was adopted and no RTF below 1 was
achieved.

| Candidate | Actual N1 evidence | Decision |
| --- | --- | --- |
| FP32 FFT output groups of four | Five fresh-process observations each: L 1.149869, candidate 1.143409 median RTF. All PCM/semantic hashes match. Decoder improves in only three of five pairs; unchanged AR also becomes faster. | Improvement too small and variable to attribute; retain L. |
| Eight-row SDOT | Forty bit-pattern checks pass with the baseline compiler policy. Some large rotating-weight shapes regress. | No full paired run or adoption. |
| NEON input quantization only | Seventy native cases plus rounding/subnormal checks pass. Full same-source synthesis calls the new kernel 33,984 times and preserves all PCM/semantic hashes. | Correctness proof only; no performance adoption. |

The final NEON full-synthesis check uses one fresh baseline and one fresh
candidate process, each with three warmups and one sample. Their archived
RTFs, 1.139587 and 1.132694, are not a repeated performance comparison and
must not be cited as an attributable speedup. Raw results, including the
initial request-guard failure and honest SHA snapshot timing, are under
`results/n1_int8_quant_neon_20261112/full_pcm/`. Subsequent wrappers compute
actual baseline library/manifest hashes before and after execution.

The integrated host suite passes 41 tests with 19 ARM-only skips; target-native
checks were executed separately on N1. The host mutex is released and no
benchmark container remains. Original service containers were preserved.
The owner subsequently approved corrected AR18+original V listening with
"V 품질 괜찮음, RTF 측정 진행". This applies to the original V400 checkpoint,
not recovery100. The verified execution manifest SHA256 is
`def34d8555ea01c87091547fccbd27aa8a3bd33e8b6bbebb0df62095004c5d7c`;
approval provenance is in `results/student_pilot/approved_v_actual18_20261112.json`.
The worker started the isolated actual-N1 L/V comparison from source `73780c6`.

## Continuing division of work

The owner explicitly requested continuing inference optimization by the same
SSH worker while the main agent handles model structure, CPU training and
quality. Model tensors/checkpoints are owned by the main agent; the worker
owns native/engine/build/measurement. The earlier bounded-three-probe stopping
condition no longer applies. Use fixed model/spec/source hashes for each
experiment and one remote compute queue; share observed shapes and costs
before requesting model changes.

The main agent started a separate 100-update AR18 QAT control from approved K,
with original AR teacher contexts and features, unchanged layer/FFN geometry,
CPU FP32 training, learning rate 2e-6, accumulation 3 and feature weight 0.1.
This tests the existing quantization-aware surrogate before considering
further pruning. It is not a native-bitwise emulator, quality approval, model
size improvement or speed result. Approved AR/V files remain unchanged.


## Accepted AR17 model and completed comparisons

The corrected V400 comparison is complete: five fresh L/V pairs give median
RTF1.160668/1.059591, with fixed18-block AR weights,118tokens and5.02s.
The owner subsequently approved the separate AR17 QAT200 + same original V400
Korean listening and requested RTF measurement. Its pinned manifest is
`e30cdb7521a7e58053c554a2d78bf6f4d187d856b13083ff2d9e15987dc85bbb`.
Five fresh AR18/AR17 pairs give medianRTF1.057507/1.012739 and token-normalized
AR16.3397/14.8897ms. Absolute wall increases5.309->5.692s because the approved
AR17 sample contains133tokens/5.62s instead of118tokens/5.02s. All measured
PCM/token/model/resource guards pass. See `results/n1_ar17_v400_20261112/`.

Runtime work now fixes approved AR17+V400 and tests only native attention
value scheduling. Model work keeps AR16, extra-recovery and inactive-FFN
candidates separate and unapproved. No model candidate silently replaces
the one being timed. Head value scheduling was adopted after exact full-model checks; see the update below.


## Exact head scheduling adopted; independent RTF remains above1

Five fixed-AR17 pairs give baseline/candidate medianRTF1.016054/0.992652,
with all five paired AR and overall ratios improved and unchanged decoder.
Candidate range0.990021..1.008962 crosses1(4/5 below). The same dim32 value
loop is adopted in canonical native source `e9e6442` without a counter.
Independent source `c356245` rebuilds the normal DSO and verifies five fresh
processes: median1.010217, range0.991603..1.017123. Therefore stable
canonical RTF<1 is not established. Every PCM/token/model/resource/library
guard passes; geometry is17blocks×16heads×dim32 and separate profile records
actual canonical head operator calls. No extra same-setting samples are added.
Raw paired/integrated evidence is `results/n1_head_v32_20261112/`.

Existing ARM workflow now exposes `studentardepth`, `studentheadv32` and
`studentheadconfirm`, all requiring warmups3/samples5. Latest canonical checks
use `studentheadconfirm` and the approved AR17 manifest. Historical causal
head comparisons must use source `75779d2`; latest native already contains
that candidate loop. The existing remote verified private-asset cache is
required; no private weights/audio/SSH material are added to Git. Workflow
options were parsed and checked without dispatching a redundant Actions run.
