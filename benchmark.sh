#!/bin/bash
#
# Run prediction speed benchmarks with recommended JVM settings.
#
# Usage:
#   ./benchmark.sh                              # run with defaults
#   ./benchmark.sh -r 20 -i 500                 # 20 measured rounds, 500 iters each
#   ./benchmark.sh -t 200 -d 15                 # 200 trees, max depth 15
#   ./benchmark.sh -r 20 -i 500 -t 200 -d 15   # all params
#
# Parameters:
#   -r  MEASURE_ROUNDS   number of measured rounds        (default: 10)
#   -i  ITERS_PER_ROUND  prediction iterations per round  (default: 400)
#   -t  NUM_TREES        number of trees in the forest    (default: 100)
#   -d  TREE_DEPTH       max tree depth (0 = unlimited)   (default: 0)

set -euo pipefail

MEASURE_ROUNDS=""
ITERS_PER_ROUND=""
NUM_TREES=""
TREE_DEPTH=""

while getopts "r:i:t:d:" opt; do
    case $opt in
        r) MEASURE_ROUNDS="$OPTARG" ;;
        i) ITERS_PER_ROUND="$OPTARG" ;;
        t) NUM_TREES="$OPTARG" ;;
        d) TREE_DEPTH="$OPTARG" ;;
        *) echo "Usage: $0 [-r rounds] [-i iters] [-t trees] [-d depth]" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

PROPS=""
[ -n "$MEASURE_ROUNDS"  ] && PROPS="$PROPS -Dbench.measureRounds=$MEASURE_ROUNDS"
[ -n "$ITERS_PER_ROUND" ] && PROPS="$PROPS -Dbench.itersPerRound=$ITERS_PER_ROUND"
[ -n "$NUM_TREES"        ] && PROPS="$PROPS -Dbench.numTrees=$NUM_TREES"
[ -n "$TREE_DEPTH"       ] && PROPS="$PROPS -Dbench.treeDepth=$TREE_DEPTH"

./gradlew benchmark $PROPS "$@"
