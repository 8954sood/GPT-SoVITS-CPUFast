#!/usr/bin/env bash
# Verify/install one immutable private bundle outside disposable source checkouts.
set -euo pipefail
umask 077
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench_root=${1:-"$HOME/gpt-sovits-benchmark"}
manifest_sha=${2:?Approved manifest SHA256 required}
action=${3:-verify}
[[ "$manifest_sha" =~ ^[0-9a-f]{64}$ ]] || exit 2
export PYTHONPATH="$repo_dir"
cache="$bench_root/student-assets/$manifest_sha"
if [[ "$action" == verify ]]; then
  [[ -f "$cache/manifest.json" ]] || exit 1
  python3 "$repo_dir/scripts/verify_student_assets.py" "$cache" "$manifest_sha"
  exit 0
fi
[[ "$action" == install ]] || exit 2
upload="$repo_dir/.private-student-upload"
python3 "$repo_dir/scripts/verify_student_assets.py" "$upload" "$manifest_sha"
if [[ -e "$cache" ]]; then
  python3 "$repo_dir/scripts/verify_student_assets.py" "$cache" "$manifest_sha"
  exit 0
fi
mkdir -p "$bench_root/student-assets"
chmod 700 "$bench_root/student-assets"
staging=$(mktemp -d "$bench_root/student-assets/.incoming.XXXXXXXX")
trap 'rm -rf "$staging"' EXIT
for name in manifest.json ar.pt decoder.pt student_spec.json; do
  cp "$upload/$name" "$staging/$name"
  chmod 600 "$staging/$name"
done
python3 "$repo_dir/scripts/verify_student_assets.py" "$staging" "$manifest_sha"
mv "$staging" "$cache"
echo 'Approved student private cache installed and verified'
