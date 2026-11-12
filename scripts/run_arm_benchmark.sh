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
student_manifest_sha=${8:-}
student_prefetch=${9:-0}
comparison_manifest_sha=${10:-}
decoder_candidate_label=${11:-R}
[[ "$decoder_candidate_label" == R || "$decoder_candidate_label" == U || "$decoder_candidate_label" == V ]] || { echo 'Invalid decoder label' >&2; exit 1; }
if [[ "$decoder_candidate_label" == V && "$mode" != studentbottleneck ]]; then echo 'V requires isolated studentbottleneck mode' >&2; exit 1; fi
[[ "$mode" == compare || "$mode" == diagnostics || "$mode" == attention || "$mode" == small || "$mode" == scheduling || "$mode" == packed48 || "$mode" == fft || "$mode" == fftcandidate || "$mode" == arcache || "$mode" == prefetch || "$mode" == prefetchcandidate || "$mode" == winograd || "$mode" == direct3 || "$mode" == locality || "$mode" == fftstream || "$mode" == transforms || "$mode" == inference || "$mode" == runtime || "$mode" == student || "$mode" == studentcompare || "$mode" == studentdecoder || "$mode" == studentheadconfirm || "$mode" == studentc24confirm || "$mode" == studentheadqk4 || "$mode" == studentheadv32 || "$mode" == studentprofile || "$mode" == studentprefillint8 || "$mode" == studentprefill || "$mode" == studentu || "$mode" == studentbottleneck || "$mode" == studentardepth || "$mode" == studentc24 || "$mode" == studentdecoderint8 || "$mode" == studentarint8 || "$mode" == studentarint8confirm || "$mode" == studenthalfstream || "$mode" == studentffthalf || "$mode" == taptiles || "$mode" == im2col ]] || { echo "Invalid benchmark mode" >&2; exit 1; }
project="$bench_root/project"
assets="$bench_root/cpufast-cefae22/assets/pretrained_models"
reference="$bench_root/reference_ja_4s.wav"
# Keep direct SSH runs and Actions runs using this script serial on the host.
exec 9>"$bench_root/.cpufast-arm-benchmark.lock"
flock -n 9 || { echo 'Another ARM benchmark holds the host lock' >&2; exit 1; }
for path in "$project/models/s1v3.ckpt" "$project/models/chinese-hubert-base" "$assets/v2Pro/s2Gv2ProPlus.pth" "$assets/sv/pretrained_eres2netv2w24s4ep4.ckpt" "$reference"; do
  if [[ ! -e "$path" ]]; then printf 'Required benchmark asset missing: %s\n' "$path" >&2; exit 1; fi
done
if [[ "$(uname -m)" != aarch64 ]]; then echo 'Target must be Linux ARM64.' >&2; exit 1; fi
if docker info >/dev/null 2>&1; then docker_cmd=(docker); else docker_cmd=(sudo -n docker); fi
"${docker_cmd[@]}" image inspect "$image" >/dev/null
mkdir -p "$repo_dir/ci-results"
student_mount=()
if [[ "$mode" == student || "$mode" == studentcompare || "$mode" == studentdecoder || "$mode" == studentheadconfirm || "$mode" == studentc24confirm || "$mode" == studentheadqk4 || "$mode" == studentheadv32 || "$mode" == studentprofile || "$mode" == studentprefillint8 || "$mode" == studentprefill || "$mode" == studentu || "$mode" == studentbottleneck || "$mode" == studentardepth || "$mode" == studentc24 || "$mode" == studentdecoderint8 || "$mode" == studentarint8 || "$mode" == studentarint8confirm || "$mode" == studenthalfstream || "$mode" == studentffthalf ]]; then
  [[ "$student_manifest_sha" =~ ^[0-9a-f]{64}$ ]] || { echo 'Approved student manifest SHA required' >&2; exit 1; }
  student_mount=(--mount "type=bind,src=$bench_root/student-assets/$student_manifest_sha,dst=/student,readonly" -e NLTK_DATA=/repo/.experiments/nltk_data)
  if [[ "$mode" == studentcompare || "$mode" == studentdecoder || "$mode" == studentu || "$mode" == studentbottleneck || "$mode" == studentardepth || "$mode" == studentc24 || "$mode" == studentdecoderint8 ]]; then
    [[ "$comparison_manifest_sha" =~ ^[0-9a-f]{64}$ ]] || { echo 'Approved comparison manifest SHA required' >&2; exit 1; }
    student_mount+=(--mount "type=bind,src=$bench_root/student-assets/$comparison_manifest_sha,dst=/student-baseline,readonly")
  fi
fi
docker_args=(run --rm --user "$(id -u):$(id -g)" --platform linux/arm64 --cpus "$cpus" --memory "$memory" --memory-swap "$memory"
  --mount "type=bind,src=$repo_dir,dst=/repo"
  --mount "type=bind,src=$assets,dst=/repo/GPT_SoVITS/pretrained_models"
  --mount "type=bind,src=$assets/v2Pro/s2Gv2ProPlus.pth,dst=/repo/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth,readonly"
  --mount "type=bind,src=$assets/sv/pretrained_eres2netv2w24s4ep4.ckpt,dst=/repo/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt,readonly"
  --mount "type=bind,src=$project,dst=/project,readonly"
  --mount "type=bind,src=$reference,dst=/reference.wav,readonly"
  "${student_mount[@]}"
  -e PYTHONPATH=/repo:/repo/GPT_SoVITS
  -e OPEN_JTALK_DICT_DIR=/project/.venv/lib/python3.12/site-packages/pyopenjtalk/open_jtalk_dic_utf_8-1.11
  -e GPT_SOVITS_DATA_DIR=/repo -e OMP_WAIT_POLICY=ACTIVE -e GOMP_SPINCOUNT=100000
  -e CPUFAST_INT8_PREFILL_LIBRARY=/repo/libcpufast_prefill_int8_probe.so -e CPUFAST_PREFILL_LIBRARY=/repo/libcpufast_prefill_probe.so -e CPUFAST_AR_LIBRARY=/repo/libcpufast_ar_torch.so -e CPUFAST_FAST_LIBRARY=/repo/libcpufast_fast.so
  "$image")
run_container() { "${docker_cmd[@]}" "${docker_args[@]}" "$@"; }
# One serial, weight-preserving FP32 comparison on the target ARM host.
# Generated audio is produced for listening; the input reference stays on the host.
mkdir -p "$repo_dir/.experiments"
if [[ "$mode" == im2col ]]; then
  run_container python resource_guard.py --report /repo/ci-results/im2col_tap_guard.json -- \
    python experiments/rtf07/bench_im2col_tap.py --output /repo/ci-results/im2col_tap.json --profile-shapes /repo/results/n1_student_l_20261112/ko_student_profile_ops.json --warmups "$warmups" --samples "$samples"
  exit 0
fi
if [[ "$mode" == taptiles ]]; then
  run_container python resource_guard.py --report /repo/ci-results/tap_tiles_guard.json -- \
    python experiments/rtf07/bench_tap_tiles.py --output /repo/ci-results/tap_tiles.json --warmups "$warmups" --samples "$samples"
  exit 0
fi
run_container python build_cpufast_ar.py --output /repo/libcpufast_ar_torch.so
run_container python -m unittest discover -s tests -p test_cpufast_tap.py -v
run_container python experiments/rtf07/test_fusion_layout.py
run_container python -m unittest discover -s tests -p test_fp32_head_attention.py -v
run_container python -m unittest discover -s tests -p test_fft_stream.py -v
run_container python -m unittest discover -s tests -p test_parallel_transforms.py -v
run_container python -m unittest discover -s tests -p test_fp32_cache_prefetch.py -v
run_container python -m unittest discover -s tests -p test_fft_channel_padding.py -v
run_container python -m unittest discover -s tests -p test_winograd_odd_channels.py -v
if [[ "$mode" == student || "$mode" == studentcompare || "$mode" == studentdecoder || "$mode" == studentheadconfirm || "$mode" == studentc24confirm || "$mode" == studentheadqk4 || "$mode" == studentheadv32 || "$mode" == studentprofile || "$mode" == studentprefillint8 || "$mode" == studentprefill || "$mode" == studentu || "$mode" == studentbottleneck || "$mode" == studentardepth || "$mode" == studentc24 || "$mode" == studentdecoderint8 || "$mode" == studentarint8 || "$mode" == studentarint8confirm || "$mode" == studenthalfstream || "$mode" == studentffthalf ]]; then
  run_container python scripts/verify_student_assets.py /student "$student_manifest_sha"
  run_container python -m nltk.downloader -q -d /repo/.experiments/nltk_data cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng
  student_args=()
  if [[ "$mode" == studentarint8 || "$mode" == studentarint8confirm || "$mode" == studentheadconfirm || "$mode" == studentc24confirm || "$mode" == studentheadqk4 || "$mode" == studentheadv32 || "$mode" == studentprofile || "$mode" == studentprefillint8 || "$mode" == studentprefill || "$mode" == studentu || "$mode" == studentbottleneck || "$mode" == studentardepth || "$mode" == studentc24 || "$mode" == studentdecoderint8 || "$mode" == studenthalfstream || "$mode" == studentffthalf ]]; then
    if [[ "$mode" == studenthalfstream ]]; then
      run_container python experiments/rtf07/build_half_stream_probe.py --output /repo/libcpufast_fast.so
      run_container python experiments/rtf07/test_half_stream_probe.py
    else
      run_container python build_cpufast_fast.py --output /repo/libcpufast_fast.so
    fi
    run_container python -m unittest discover -s tests -p test_fast_operators.py -v
    student_args=(--ar-int8-comparison)
    if [[ "$mode" == studentffthalf ]]; then
      run_container python -m unittest discover -s tests -p test_fft_half_runtime.py -v
      student_args=(--fft-half-comparison)
    fi
    if [[ "$mode" == studenthalfstream ]]; then student_args=(--fft-half-stream-comparison); fi
    if [[ "$mode" == studentprefill ]]; then
      run_container python experiments/rtf07/build_prefill_probe.py --output /repo/libcpufast_prefill_probe.so
      run_container python experiments/rtf07/test_prefill_probe.py
      student_args=(--packed-prefill-comparison)
    fi
    if [[ "$mode" == studentprefillint8 ]]; then
      run_container python experiments/rtf07/build_prefill_int8_probe.py --output /repo/libcpufast_prefill_int8_probe.so
      run_container python experiments/rtf07/test_prefill_int8_probe.py
      student_args=(--int8-prefill-comparison)
    fi
    if [[ "$mode" == studentheadconfirm ]]; then student_args=(--head-integrated-confirm); fi
    if [[ "$mode" == studentc24confirm ]]; then student_args=(--c24-integrated-confirm); fi
    if [[ "$mode" == studentheadqk4 ]]; then
      run_container python experiments/rtf07/build_head_qk4_probe.py --output /repo/libcpufast_head_qk4_runtime.so --replace-runtime
      student_args=(--head-qk4-comparison)
    fi
    if [[ "$mode" == studentheadv32 ]]; then
      run_container python experiments/rtf07/build_head_v32_probe.py --output /repo/libcpufast_head_v32_runtime.so --replace-runtime
      student_args=(--head-v32-comparison)
    fi
    if [[ "$mode" == studentprofile ]]; then student_args=(--diagnostic-profile-only); fi
    if [[ "$mode" == studentarint8confirm ]]; then student_args+=(--ar-int8-confirm-only); fi
  fi

  if [[ "$mode" == studentcompare || "$mode" == studentdecoder || "$mode" == studentu || "$mode" == studentbottleneck || "$mode" == studentardepth || "$mode" == studentc24 || "$mode" == studentdecoderint8 ]]; then
    run_container python scripts/verify_student_assets.py /student-baseline "$comparison_manifest_sha"
    student_args=(--baseline-manifest /student-baseline/manifest.json --baseline-manifest-sha256 "$comparison_manifest_sha" --prefetch "$student_prefetch")
    if [[ "$mode" == studentardepth ]]; then student_args+=(--ar-depth-comparison); fi
    if [[ "$mode" == studentdecoder || "$mode" == studentu || "$mode" == studentbottleneck || "$mode" == studentdecoderint8 ]]; then student_args+=(--decoder-only); fi
    if [[ "$mode" == studentdecoderint8 || "$mode" == studentu || "$mode" == studentbottleneck ]]; then student_args+=(--decoder-ar-int8); fi
    if [[ "$mode" == studentu || "$mode" == studentbottleneck ]]; then
      run_container python -m unittest discover -s tests -p test_rectangular_bottleneck_runtime.py -v
      student_args+=(--alternating-decoder-comparison)
      if [[ "$mode" == studentu ]]; then student_args+=(--decoder-candidate-label U); fi
      if [[ "$mode" == studentbottleneck ]]; then student_args+=(--decoder-candidate-label "$decoder_candidate_label"); fi
    fi
  fi
  if [[ "$mode" == studentc24 ]]; then
    student_args+=(--decoder-only --decoder-ar-int8 --alternating-decoder-comparison --c24-model-comparison)
  fi
  run_container python scripts/run_student_matrix.py --manifest /student/manifest.json \
    --manifest-sha256 "$student_manifest_sha" --output /repo/ci-results --warmups "$warmups" --samples "$samples" "${student_args[@]}"
  if [[ "$mode" == studentarint8confirm ]]; then
    run_container python resource_guard.py --report /repo/ci-results/im2col_tap_guard.json -- \
      python experiments/rtf07/bench_im2col_tap.py --output /repo/ci-results/im2col_tap.json --profile-shapes /repo/ci-results/ko_student_profile_ops.json --warmups "$warmups" --samples "$samples"
  fi
  if [[ "$mode" == studentdecoder ]]; then
    run_container python resource_guard.py --report /repo/ci-results/tap_tiles_guard.json -- \
      python experiments/rtf07/bench_tap_tiles.py --output /repo/ci-results/tap_tiles.json --warmups "$warmups" --samples "$samples" --profile-shapes /repo/ci-results/ko_student_profile_ops.json
  fi
  exit 0
fi
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
if [[ "$mode" == runtime ]]; then
  for index in 0 1; do
    number=$((index + 1))
    if (( index == 0 )); then variants=(baseline parallel inference combined); else variants=(baseline combined inference parallel); fi
    for variant in "${variants[@]}"; do
      candidate_args=()
      compare_args=()
      if [[ "$variant" != baseline ]]; then compare_args=(--compare-pcm "/repo/ci-results/baseline_${number}.wav"); fi
      if [[ "$variant" == parallel ]]; then candidate_args=(--parallel-transforms); fi
      if [[ "$variant" == inference ]]; then candidate_args=(--inference-mode); fi
      if [[ "$variant" == combined ]]; then candidate_args=(--fft-stream --parallel-transforms --inference-mode); fi
      run_container python resource_guard.py --report "/repo/ci-results/${variant}_${number}_guard.json" -- \
        python benchmark_cpufast.py "${common[@]}" --tap-small "${candidate_args[@]}" --text "${texts[$index]}" \
        "${compare_args[@]}" --output "/repo/ci-results/${variant}_${number}.json"
    done
  done
  exit 0
fi
for index in 0 1; do
  number=$((index + 1))
  if [[ "$mode" == attention || "$mode" == packed48 || "$mode" == fftcandidate || "$mode" == arcache || "$mode" == prefetch || "$mode" == prefetchcandidate || "$mode" == winograd || "$mode" == direct3 || "$mode" == locality || "$mode" == fftstream || "$mode" == transforms || "$mode" == inference || "$mode" == runtime || "$mode" == student || "$mode" == studentcompare || "$mode" == studentdecoder || "$mode" == studentheadconfirm || "$mode" == studentc24confirm || "$mode" == studentheadqk4 || "$mode" == studentheadv32 || "$mode" == studentprofile || "$mode" == studentprefillint8 || "$mode" == studentprefill || "$mode" == studentu || "$mode" == studentbottleneck || "$mode" == studentardepth || "$mode" == studentc24 || "$mode" == studentdecoderint8 || "$mode" == studentarint8 || "$mode" == studentarint8confirm || "$mode" == studenthalfstream || "$mode" == studentffthalf ]]; then
    candidate_args=(--native-attention)
    if [[ "$mode" == packed48 ]]; then candidate_args=(--packed-small-48); fi
    if [[ "$mode" == fftcandidate ]]; then candidate_args=(--n1-fft-sizes); fi
    if [[ "$mode" == arcache ]]; then candidate_args=(--ar-fp32-cache); fi
    if [[ "$mode" == prefetchcandidate ]]; then candidate_args=(--ar-prefetch 256); fi
    if [[ "$mode" == direct3 ]]; then candidate_args=(--n1-small-direct); fi
    if [[ "$mode" == fftstream ]]; then candidate_args=(--fft-stream); fi
    if [[ "$mode" == transforms ]]; then candidate_args=(--parallel-transforms); fi
    if [[ "$mode" == inference ]]; then candidate_args=(--inference-mode); fi
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
