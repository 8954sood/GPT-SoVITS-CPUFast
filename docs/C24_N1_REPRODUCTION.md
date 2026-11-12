# Approved AR17/C24 model reproduction

The owner approved Korean audio from AR17 QAT200 plus the final-C24 two-branch
Vtarget100 decoder. The reference, text and seed42 are fixed by each approved
manifest. This does not approve another model, another decoder recovery, or
Japanese/English listening quality.

| Bundle | Manifest SHA256 |
| --- | --- |
| Original approved AR17/V400 baseline | `e30cdb7521a7e58053c554a2d78bf6f4d187d856b13083ff2d9e15987dc85bbb` |
| Approved AR17/C24 Vtarget100 | `696966debccf516f2af6ef0f87be2c2cf52548d313d70f8f467731dbf1f7a865` |

Both use identical AR bytes `a25ae9826d4d608ca41adca04075ff75a5ac66ababa0e823da60b7990f354352`.
C24 decoder SHA is `afac9da039331cb1a8511d709998396a28ffc29469ba092fda3c7fc823d0ab11`;
spec SHA is `3a6550930e73dafadd07a9fdbe79a4e00fe7a55e1bfcd370ff237d0da074dc69`.
Its approved Korean PCM SHA is `ab9179ec076b5b3e1bd7f90879486c4e4447d183936f9ae698bc714260a6b279`.
Both produce133semantic tokens and5.62seconds on this request.

Actual N1 measured results: five counterbalanced C24 samples plus five
independent fresh confirmations all stayed below1. Their complete range was
.934545–.962389, median.945295 and worst.962389. Raw reports and excluded
profiles are retained in `results/n1_c24_vtarget100_20261112`; no slow sample
was removed or replaced. These are warm nonstreaming Korean results on the
fixed reference/text/seed and resource limits below.

## Direct SSH host

Use the existing private assets, image and verified immutable student cache
layout established by the environment handoff. These commands run inside a disposable
source checkout on the ARM host. They require the approved four-file bundles
under `$BENCH_ROOT/student-assets/<manifestSHA>/`. No new C24/AR17 model weights, SSH key or private reference audio are added
to source Git.

```bash
bash scripts/run_arm_benchmark.sh /home/ubuntu/gpt-sovits-benchmark \
  cpufast-n1:cefae22 8000000000 2 3 5 studentc24 \
  696966debccf516f2af6ef0f87be2c2cf52548d313d70f8f467731dbf1f7a865 256 \
  e30cdb7521a7e58053c554a2d78bf6f4d187d856b13083ff2d9e15987dc85bbb

bash scripts/run_arm_benchmark.sh /home/ubuntu/gpt-sovits-benchmark \
  cpufast-n1:cefae22 8000000000 2 3 5 studentc24confirm \
  696966debccf516f2af6ef0f87be2c2cf52548d313d70f8f467731dbf1f7a865 256
```

Use a fresh checkout/output directory for each invocation. The comparison is
exactly five fresh process pairs in alternating AB/BA order; each process has
three excluded warmups and one timed sample. Confirmation is a separate fixed
five-process experiment on the approved C24 bundle. An additional operator
profile is retained separately and excluded from performance aggregates. The
RTF measures warm nonstreaming utterance inference, with load, warmups and WAV
disk writing excluded. It does not measure streaming TTFA.

Both modes preserve canonical value scheduling, original Tap epilogues,
Cin240 FFT256 and normal FP32 acoustics. They do not enable the rejected
FFT128/QK4 candidates or the unadopted FFN2/Tap-finish probes. The comparison
allows only the final-stage original K11 branch removal, same AR17 bytes and
other shared geometry, and verifies actual12/11 residual blocks. Every C24
warmup/sample must reproduce its own approved semantic, PCM and duration.
Cgroups must report2vCPU,8000000000memory bytes and zero swap; original/student
file and parameter guards must pass. Confirmation also checks actual canonical
head geometry/operator and native DSO/manifest hashes before and after.

## GitHub Actions

Select `studentc24` or `studentc24confirm` in `.github/workflows/arm-rtf.yml`,
with warmups3 and samples5. Supply the candidate manifest SHA above; comparison
also needs the baseline manifest SHA. The host cache is verified first by
trusted manifest SHA and all file hashes. On a cache miss, provide the private
HF repository, full immutable revision and approved manifest paths so the
existing secret-backed downloader can install identical files. No model files
or credentials are added to public Git. The workflow reports every raw RTF and
its maximum; a median below1 is insufficient for the slowest-sample goal.
