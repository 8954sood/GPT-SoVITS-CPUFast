#!/usr/bin/env bash
set -euo pipefail
root=/home/ubuntu/gpt-sovits-benchmark
repo=$root/ssh-c24-model-24e562a-20261112
canonical=$root/ssh-head-integrated-c356245-20261112
manifest_sha=${1:?Approved immutable manifest required}
exec 9>"$root/.cpufast-arm-benchmark.lock"
flock -n 9 || { echo 'ARM host busy'; exit 1; }
cp "$canonical"/libcpufast_{ar_torch,fast}.so* "$repo/"
mkdir -p "$repo/.experiments" "$repo/ci-results"
cp -a "$canonical/.experiments/nltk_data" "$repo/.experiments/"
args=(run --rm --user "$(id -u):$(id -g)" --platform linux/arm64 --cpus 2 --memory 8000000000 --memory-swap 8000000000
 --mount "type=bind,src=$repo,dst=/repo"
 --mount "type=bind,src=$root/cpufast-cefae22/assets/pretrained_models,dst=/repo/GPT_SoVITS/pretrained_models"
 --mount "type=bind,src=$root/project,dst=/project,readonly"
 --mount "type=bind,src=$root/reference_ja_4s.wav,dst=/reference.wav,readonly"
 --mount "type=bind,src=$root/student-assets/$manifest_sha,dst=/student,readonly"
 --mount "type=bind,src=$root/student-assets/e30cdb7521a7e58053c554a2d78bf6f4d187d856b13083ff2d9e15987dc85bbb,dst=/baseline,readonly"
 -e PYTHONPATH=/repo:/repo/GPT_SoVITS
 -e OPEN_JTALK_DICT_DIR=/project/.venv/lib/python3.12/site-packages/pyopenjtalk/open_jtalk_dic_utf_8-1.11
 -e GPT_SOVITS_DATA_DIR=/repo -e NLTK_DATA=/repo/.experiments/nltk_data
 -e OMP_WAIT_POLICY=ACTIVE -e GOMP_SPINCOUNT=100000
 -e CPUFAST_AR_LIBRARY=/repo/libcpufast_ar_torch.so -e CPUFAST_FAST_LIBRARY=/repo/libcpufast_fast.so
 cpufast-n1:cefae22)
run() { sudo -n docker "${args[@]}" "$@"; }
run python scripts/verify_student_assets.py /student "$manifest_sha"
run python scripts/verify_student_assets.py /baseline e30cdb7521a7e58053c554a2d78bf6f4d187d856b13083ff2d9e15987dc85bbb
run python scripts/run_student_matrix.py --manifest /student/manifest.json --manifest-sha256 "$manifest_sha" --baseline-manifest /baseline/manifest.json --baseline-manifest-sha256 e30cdb7521a7e58053c554a2d78bf6f4d187d856b13083ff2d9e15987dc85bbb --output /repo/ci-results --warmups 3 --samples 5 --decoder-only --decoder-ar-int8 --alternating-decoder-comparison --c24-model-comparison
