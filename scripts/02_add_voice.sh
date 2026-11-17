#!/usr/bin/env bash
# Register a new cloned voice, poll the job and save the voice_id for 04_tts.sh. Usage: 02_add_voice.sh [sample]
# The sample names <sample>.wav and <sample>.txt in INPUTS_DIR and defaults to jessica. Every run adds a new voice.
set -euo pipefail
source "$(dirname "$0")/lib/mcp.sh"
require_tools

DEFAULT_SAMPLE=jessica
sample="${1:-$DEFAULT_SAMPLE}"
VOICE_REFERENCE="${VOICE_REFERENCE:-$INPUTS_DIR/$sample.wav}"
VOICE_TRANSCRIPT="${VOICE_TRANSCRIPT:-${VOICE_REFERENCE%.*}.txt}"
VOICE_NAME="${VOICE_NAME:-$(printf '%s' "${sample:0:1}" | tr '[:lower:]' '[:upper:]')${sample:1}}"
VOICE_LANGUAGE="${VOICE_LANGUAGE:-en}"

voice_status() { jq -r --arg id "$2" '.voices[] | select(.voice_id == $id) | .status' <<<"$1"; }

[[ -f "$VOICE_REFERENCE" ]] || die "no reference recording at $VOICE_REFERENCE"
[[ -f "$VOICE_TRANSCRIPT" ]] || die "no transcript at $VOICE_TRANSCRIPT"
if [[ "$VOICE_REFERENCE" != "$INPUTS_DIR/$DEFAULT_SAMPLE.wav" && "${VOICE_CONSENT:-}" != yes ]]; then
  die "set VOICE_CONSENT=yes to confirm you are authorized to clone the voice in $VOICE_REFERENCE"
fi

arguments=$(jq -cn --arg name "$VOICE_NAME" --arg path "$VOICE_REFERENCE" --arg language "$VOICE_LANGUAGE" \
  --rawfile text "$VOICE_TRANSCRIPT" '{name: $name, reference_audio_path: $path, reference_text: ($text | rtrimstr("\n")),
  language: $language, consent_confirmed: true, wait: false}')

mcp_connect
submission=$(mcp_tool register_voice "$arguments") || die "register_voice was rejected: $submission"
voice_id=$(jq -r '.voice_id // empty' <<<"$submission")
[[ -n "$voice_id" ]] || die "register_voice returned no voice_id: $submission"
pass "submitted registration of voice $VOICE_NAME ($voice_id)"

mcp_poll_job "$(job_id_of "$submission")" >/dev/null || die "voice registration did not complete"
voices=$(mcp_tool list_voices) || die "list_voices failed: $voices"
[[ "$(voice_status "$voices" "$voice_id")" == ready ]] || die "voice $voice_id is not ready"
state_set voice_id "$voice_id"
pass "voice $VOICE_NAME ($voice_id) is ready and saved for 04_tts.sh"
