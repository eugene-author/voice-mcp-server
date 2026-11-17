#!/usr/bin/env bash
# Transcribe a directory, poll the batch and list its child jobs. Usage: 06_batch.sh [directory]
set -euo pipefail
source "$(dirname "$0")/lib/mcp.sh"
require_tools

directory="$(abs_path "${1:-$REPO_ROOT/var/inputs/postman-batch}")"
[[ -d "$directory" ]] || die "no directory at $directory"

mcp_connect
submission=$(mcp_tool transcribe_batch "$(jq -cn --arg directory "$directory" '{directory: $directory, wait: false}')") ||
  die "transcribe_batch was rejected: $submission"
job_id=$(job_id_of "$submission")
pass "submitted batch $job_id for $directory"

view=$(mcp_poll_job "$job_id") || die "the batch did not complete"
log "counts: $(jq -c '.counts // {}' <<<"$view")"
children=$(mcp_tool list_jobs "$(jq -cn --arg id "$job_id" '{parent_job_id: $id}')") || die "list_jobs failed: $children"
total=$(jq '.items | length' <<<"$children")
completed=$(jq '[.items[] | select(.status == "completed")] | length' <<<"$children")
((total > 0 && total == completed)) || die "$completed of $total child jobs completed"
pass "batch completed with $completed of $total child jobs"
