#!/bin/bash

# Run FasterForest unit tests.
# Requires: JDK 22+

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/gradlew" -p "$SCRIPT_DIR" test "$@"
