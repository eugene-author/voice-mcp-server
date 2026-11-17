#!/usr/bin/env bash
# Run the tests in order and stop on the first failure. Usage: run_all.sh [--all]
# The default runs 01 to 04; --all also runs the batch test.
set -uo pipefail
cd "$(dirname "$0")"

tests=(01_connect.sh 02_add_voice.sh 03_stt.sh 04_tts.sh)
[[ "${1:-}" == --all ]] && tests+=(05_batch.sh)
summary=()
failed=0

for test in "${tests[@]}"; do
  printf '\n===== %s =====\n' "$test" >&2
  started=$(date +%s)
  if ./"$test"; then
    result=PASS
  else
    result=FAIL
    failed=1
  fi
  summary+=("$(printf '%-5s %-26s %ss' "$result" "$test" "$(($(date +%s) - started))")")
  ((failed == 0)) || break
done

printf '\n===== summary =====\n' >&2
printf '%s\n' "${summary[@]}" >&2
exit "$failed"
