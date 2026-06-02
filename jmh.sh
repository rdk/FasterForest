#!/bin/bash
#
# Run the JMH prediction-speed microbenchmarks.
#
# This is the JMH counterpart of benchmark.sh (which runs the legacy
# hand-rolled PredictionSpeedBenchmark). Both harnesses are kept side by side.
#
# Usage:
#   ./jmh.sh                                  # run all benchmarks with defaults
#   ./jmh.sh -b ForestPredict                 # only the heap-forest benchmark
#   ./jmh.sh -b NativeZeroCopy                 # only the native zero-copy benchmark
#   ./jmh.sh -t Flat                          # only forestType=Flat
#   ./jmh.sh -n 69500                          # tile batch to 69500 rows
#   ./jmh.sh -b ForestPredictBenchmark.batchPredict -n 6950,69500,695000   # batch-size sweep
#   ./jmh.sh -f 3 -wi 5 -i 10                  # 3 forks, 5 warmup, 10 measured iters
#   ./jmh.sh -- -prof gc ForestPredict         # pass raw JMH args after --
#
# Parameters:
#   -b  BENCH_REGEX      benchmark name filter (e.g. ForestPredict, NativeZeroCopy)
#   -t  FOREST_TYPE      restrict to one forestType param (e.g. Flat, NativePanama)
#   -n  BATCH_SIZE       batch rows, tiled from dataset (comma-list to sweep)
#   -f  FORKS            number of JVM forks                    (JMH default: 1)
#   -wi WARMUP_ITERS     warmup iterations                      (bench default: 5)
#   -i  MEASURE_ITERS    measured iterations                    (bench default: 8)
#   --                   everything after this is passed verbatim to JMH
#
set -euo pipefail

JMH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b)   JMH_ARGS+=("$2"); shift 2 ;;
        -t)   JMH_ARGS+=("-p" "forestType=$2"); shift 2 ;;
        -n)   JMH_ARGS+=("-p" "batchSize=$2"); shift 2 ;;
        -f)   JMH_ARGS+=("-f" "$2"); shift 2 ;;
        -wi)  JMH_ARGS+=("-wi" "$2"); shift 2 ;;
        -i)   JMH_ARGS+=("-i" "$2"); shift 2 ;;
        --)   shift; JMH_ARGS+=("$@"); break ;;
        *)    echo "Unknown option: $1" >&2
              echo "Usage: $0 [-b regex] [-t forestType] [-n batchSize] [-f forks] [-wi warmup] [-i iters] [-- raw JMH args]" >&2
              exit 1 ;;
    esac
done

if [ ${#JMH_ARGS[@]} -eq 0 ]; then
    ./gradlew jmh
else
    ./gradlew jmh -PjmhArgs="${JMH_ARGS[*]}"
fi
