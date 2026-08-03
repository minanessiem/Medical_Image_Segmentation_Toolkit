#!/usr/bin/env bash
set -euo pipefail

python3 -m scripts.gc_submission_builder.cli save "$@"
