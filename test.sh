#!/bin/bash

# Run FasterForest unit tests (excludes benchmarks).
# Requires: JDK 22+
#
# Usage:
#   ./test.sh                                   # run all tests
#   ./test.sh --tests "*.FasterForestTest"       # run specific test class

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/gradlew" -p "$SCRIPT_DIR" cleanTest test "$@"
