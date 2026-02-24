#!/usr/bin/env bash
# Chained formatter: runs ruff format (or black) on stdin, then pipes through joy.
# Usage in VSCode Custom Local Formatters:
#   "command": "/path/to/joy_formatter/format_chain.sh"
#
# To switch to black, replace the ruff line with:
#   black --quiet -
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ruff format --stdin-filename "${1:-stdin.py}" - | python "$SCRIPT_DIR/joy.py"
