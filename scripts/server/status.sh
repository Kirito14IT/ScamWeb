#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
exec "$VENV_PYTHON" -m scammark.run_status \
  --config "$CONFIG_PATH" --tmux-session "$TMUX_SESSION"
