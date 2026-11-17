#!/bin/sh
# Optionally apply migrations, then run the CLI (default: serve).
set -eu
if [ "${MIGRATE_ON_START:-false}" = "true" ] && [ "${1:-serve}" = "serve" ]; then
    python -m app migrate
fi
exec python -m app "$@"
