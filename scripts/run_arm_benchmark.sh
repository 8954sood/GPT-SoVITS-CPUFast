#!/usr/bin/env bash
# Run in a private, disposable source checkout on the target Linux ARM64 host.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench_root=${1:-"$HOME/gpt-sovits-benchmark"}
image=${2:-cpufast-n1:cefae22}
memory=${3:-8000000000}
cpus=${4:-2}
warmups=${5:-2}
samples=${6:-3}
mode=${7:-compare}
[[ "$mode" == compare || "$mode" == diagnostics || "$mode" == attention || "$mode" == small || "$mode" == scheduling || "$mode" == packed48 || "$mode" == fft || "$mode" == fftcandidate || "$mode" == arcache || "$mode" == prefetch || "$mode" == prefetchcandidate || "$mode" == winograd || "$mode" == direct3 || "$mode" == locality || "$mode" == fftstream || "$mode" == transforms ]] || { echo "Invalid benchmark mode" >&2; exit 1; }
project="$bench_root/project"
assets="$bench_root/cpufast-cefae22/assets/pretrained_models"
reference="$bench_root/reference_ja_4s.wav"
for path in "$project/models/s1v3.ckpt" "$project/models/chinese-hubert-base" "$assets/v2Pro/s2Gv2ProPlus.pth" "$assets/sv/pretrained_eres2netv2w24s4ep4.ckpt" "$reference"; do
  if [[ ! -e "$path" ]]; then printf 'Required benchmark asset missing: %s\n' "$path" >&2; exit 1; fi
done
if [[ "$(uname -m)" != aarch64 ]]; then echo 'Target must be Linux ARM64.' >&2; exit 1; fi
if docker info >/dev/null 2>&1; then docker_cmd=(docker); else docker_cmd=(sudo -n docker); fi
"${docker_cmd[@]}" image inspect "$image" >/dev/null
mkdir -p "$repo_dir/ci-results"
docker_args=(run --rm --user "$(id -u):$(id -g)" --platform linux/arm64 --cpus "$cpus" --memory "$memory" --memory-swap "$memory"
  --mount "type=bind,src=$repo_dir,dst=/repo"
  --mount "type=bind,src=$assets,dst=/repo/GPT_SoVITS/pretrained_models"
  --mount "type=bind,src=$assets/v2Pro/s2Gv2ProPlus.pth,dst=/repo/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth,readonly"
  --mount "type=bind,src=$assets/sv/pretrained_eres2netv2w24s4ep4.ckpt,dst=/repo/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt,readonly"
  --mount "type=bind,src=$project,dst=/project,readonly"
  --mount "type=bind,src=$reference,dst=/reference.wav,readonly"
  -e PYTHONPATH=/repo:/repo/GPT_SoVITS
  -e OPEN_JTALK_DICT_DIR=/project/.venv/lib/python3.12/site-packages/pyopenjtalk/open_jtalk_dic_utf_8-1.11
  -e GPT_SOVITS_DATA_DIR=/repo -e OMP_WAIT_POLICY=ACTIVE -e GOMP_SPINCOUNT=100000
  -e CPUFAST_AR_LIBRARY=/repo/libcpufast_ar_torch.so -e CPUFAST_FAST_LIBRARY=/repo/libcpufast_fast.so
  "$image")
run_container() { "${docker_cmd[@]}" "${docker_args[@]}" "$@"; }
# One serial, weight-preserving FP32 comparison on the target ARM host.
# Generated audio is produced for listening; the input reference stays on the host.
mkdir -p "$repo_dir/.experiments"
run_container python build_cpufast_ar.py --output /repo/libcpufast_ar_torch.so
run_container python -m unittest discover -s tests -p test_cpufast_tap.py -v
run_container python experiments/rtf07/test_fusion_layout.py
run_container python -m unittest discover -s tests -p test_fp32_head_attention.py -v
run_container python -m unittest discover -s tests -p test_fft_stream.py -v
run_container python -m unittest discover -s tests -p test_parallel_transforms.py -v
common=(--engine --head-major --cook-phase --fft-layout --packed-fft --fused-finish
  --ar-threads 2 --threads 2
  --s1 /project/models/s1v3.ckpt --s2 /repo/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth
  --hubert /project/models/chinese-hubert-base --reference /reference.wav
  --prompt-text 'スマートフォンからフィーチャーフォンまで、マルチデバイスに対応。'
  --warmups "$warmups" --samples "$samples")
texts=('今日は穏やかな天気ですね。少し休んでから、また一緒に続きを進めましょう。'
       '今日は新しい音声合成モデルを使って、声の雰囲気を確かめています。')
if [[ "$mode" == prefetchcandidate ]]; then common+=(--ar-prefetch 0); fi
if [[ "$mode" == locality ]]; then
  run_container python resource_guard.py --report /repo/ci-results/ar_locality_guard.json -- \
    python experiments/rtf07/bench_ar_locality.py --output /repo/ci-results/ar_locality.json
  exit 0
fi
if [[ "$mode" == winograd ]]; then
  run_container python resource_guard.py --report /repo/ci-results/winograd_methods_guard.json -- \
    python experiments/rtf07/bench_winograd_methods.py --output /repo/ci-results/winograd_methods.json
  exit 0
fi
if [[ "$mode" == prefetch ]]; then
  run_container python resource_guard.py --report /repo/ci-results/ar_prefetch_guard.json -- \
    python experiments/rtf07/bench_ar_prefetch.py --output /repo/ci-results/ar_prefetch.json
  exit 0
fi
if [[ "$mode" == fft ]]; then
  run_container python resource_guard.py --report /repo/ci-results/fft_sizes_guard.json -- \
    python experiments/rtf07/bench_fft_sizes.py --output /repo/ci-results/fft_sizes.json
  exit 0
fi
if [[ "$mode" == scheduling ]]; then
  for policy in active passive; do
    if [[ "$policy" == active ]]; then wait_policy=ACTIVE; spins=100000; else wait_policy=PASSIVE; spins=0; fi
    compare_args=()
    if [[ "$policy" == passive ]]; then compare_args=(--compare-pcm /repo/ci-results/active.wav); fi
    run_container env OMP_WAIT_POLICY="$wait_policy" GOMP_SPINCOUNT="$spins" \
      python resource_guard.py --report "/repo/ci-results/${policy}_guard.json" -- \
      python benchmark_cpufast.py "${common[@]}" --tap-small --text "${texts[0]}" \
      "${compare_args[@]}" --output "/repo/ci-results/${policy}.json"
  done
  exit 0
fi
if [[ "$mode" == small ]]; then
  run_container python resource_guard.py --report /repo/ci-results/decoder_small_guard.json -- \
    python experiments/rtf07/bench_decoder_small.py --output /repo/ci-results/decoder_small.json
  exit 0
fi
if [[ "$mode" == fftstream ]]; then
  run_container python resource_guard.py --report /repo/ci-results/fft_stream_guard.json -- \
    python experiments/rtf07/bench_fft_stream.py --output /repo/ci-results/fft_stream.json
fi
if [[ "$mode" == diagnostics ]]; then
  run_container python resource_guard.py --report /repo/ci-results/profile_guard.json -- \
    python benchmark_cpufast.py "${common[@]}" --tap-small --text "${texts[0]}" \
    --output /repo/ci-results/profile_run.json --profile-output /repo/ci-results/profile_ops.json
  for probe in ar_gemv head_attention decoder_methods; do
    echo "Running synthetic FP32 probe: $probe"
    run_container python resource_guard.py --report "/repo/ci-results/${probe}_guard.json" -- \
      python "experiments/rtf07/bench_${probe}.py" --output "/repo/ci-results/${probe}.json"
  done
  exit 0
fi
for index in 0 1; do
  number=$((index + 1))
  if [[ "$mode" == attention || "$mode" == packed48 || "$mode" == fftcandidate || "$mode" == arcache || "$mode" == prefetch || "$mode" == prefetchcandidate || "$mode" == winograd || "$mode" == direct3 || "$mode" == locality || "$mode" == fftstream || "$mode" == transforms ]]; then
    candidate_args=(--native-attention)
    if [[ "$mode" == packed48 ]]; then candidate_args=(--packed-small-48); fi
    if [[ "$mode" == fftcandidate ]]; then candidate_args=(--n1-fft-sizes); fi
    if [[ "$mode" == arcache ]]; then candidate_args=(--ar-fp32-cache); fi
    if [[ "$mode" == prefetchcandidate ]]; then candidate_args=(--ar-prefetch 256); fi
    if [[ "$mode" == direct3 ]]; then candidate_args=(--n1-small-direct); fi
    if [[ "$mode" == fftstream ]]; then candidate_args=(--fft-stream); fi
    if [[ "$mode" == transforms ]]; then candidate_args=(--parallel-transforms); fi
    echo "Benchmarking sentence $number: approved tap and $mode candidate"
    run_container python resource_guard.py --report "/repo/ci-results/tap_${number}_guard.json" -- \
      python benchmark_cpufast.py "${common[@]}" --tap-small --text "${texts[$index]}" \
      --output "/repo/ci-results/tap_${number}.json"
    profile_args=()
    if [[ "$mode" == attention && "$number" == 1 ]]; then profile_args=(--profile-output /repo/ci-results/attention_profile_ops.json); fi
    run_container python resource_guard.py --report "/repo/ci-results/attention_${number}_guard.json" -- \
      python benchmark_cpufast.py "${common[@]}" --tap-small "${candidate_args[@]}" --text "${texts[$index]}" \
      --compare-pcm "/repo/ci-results/tap_${number}.wav" "${profile_args[@]}" \
      --output "/repo/ci-results/attention_${number}.json"
    continue
  fi
  echo "Benchmarking sentence $number: FP32 baseline"
  run_container python resource_guard.py --report "/repo/ci-results/baseline_${number}_guard.json" -- \
    python benchmark_cpufast.py "${common[@]}" --text "${texts[$index]}" \
    --output "/repo/ci-results/baseline_${number}.json"
  echo "Benchmarking sentence $number: FP32 tap candidate"
  run_container python resource_guard.py --report "/repo/ci-results/tap_${number}_guard.json" -- \
    python benchmark_cpufast.py "${common[@]}" --tap-small --text "${texts[$index]}" \
    --compare-pcm "/repo/ci-results/baseline_${number}.wav" \
    --output "/repo/ci-results/tap_${number}.json"
done

if [[ "$mode" == fftcandidate ]]; then
  run_container python resource_guard.py --report /repo/ci-results/ar_cache_guard.json -- \
    python experiments/rtf07/bench_ar_cache.py --output /repo/ci-results/ar_cache.json
fi
