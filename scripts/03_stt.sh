#!/usr/bin/env bash
# Transcribe a file as a background job and poll it. Usage: 03_stt.sh [file]
# Optional: STT_MODEL, STT_LANGUAGE, STT_PRESET, STT_DENOISE=true.
set -euo pipefail
source "$(dirname "$0")/lib/mcp.sh"
require_tools

file="$(abs_path "${1:-$INPUTS_DIR/jessica.wav}")"
[[ -f "$file" ]] || die "no audio file at $file"

arguments=$(jq -cn --arg file "$file" --arg model "${STT_MODEL:-}" --arg language "${STT_LANGUAGE:-}" \
  --arg preset "${STT_PRESET:-}" --arg denoise "${STT_DENOISE:-}" '{file_path: $file, wait: false}
  + (if $model != "" then {model: $model} else {} end)
  + (if $language != "" then {language: $language} else {} end)
  + (if $preset != "" then {preset: $preset} else {} end)
  + (if $denoise != "" then {options: {denoise: ($denoise == "true")}} else {} end)')

mcp_connect
submission=$(mcp_tool speech_to_text "$arguments") || die "speech_to_text was rejected: $submission"
pass "submitted speech_to_text for $(basename "$file")"

view=$(mcp_poll_job "$(job_id_of "$submission")") || die "transcription did not complete"
text=$(jq -r '.text // empty' <<<"$view")
[[ -n "$text" ]] || die "the job completed without a transcript"
log "output files:"
files=$(check_output_files "$view")
pass "transcription completed with $files output files"
log "transcript: $text"
