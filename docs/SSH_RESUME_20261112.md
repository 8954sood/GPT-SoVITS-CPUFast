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
