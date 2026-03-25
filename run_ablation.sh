#!/bin/bash
# run_ablation.sh — Compiler flag ablation study
# Tests the impact of different compiler optimizations on GBDT training
# Usage: chmod +x run_ablation.sh && ./run_ablation.sh

set -euo pipefail

SOURCE="gboost_neon.cpp"
RESULTS="ablation_results.txt"

# Auto-detect chip
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple M1")
if echo "$CHIP" | grep -q "M3"; then
    MCPU="apple-m3"
elif echo "$CHIP" | grep -q "M2"; then
    MCPU="apple-m2"
elif echo "$CHIP" | grep -q "M4"; then
    MCPU="apple-m4"
else
    MCPU="apple-m1"
fi

echo "============================================================" | tee "$RESULTS"
echo "  COMPILER FLAG ABLATION STUDY" | tee -a "$RESULTS"
echo "  $(date)" | tee -a "$RESULTS"
echo "  CPU: $CHIP (using -mcpu=$MCPU)" | tee -a "$RESULTS"
echo "  $(uname -m) | $(sw_vers -productName) $(sw_vers -productVersion)" | tee -a "$RESULTS"
echo "  $(clang++ --version | head -1)" | tee -a "$RESULTS"
echo "============================================================" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# Configurations to test (uses auto-detected MCPU)
declare -a CONFIGS=(
    "-O0"
    "-O1"
    "-O2"
    "-O3"
    "-Os"
    "-Oz"
    "-O3 -mcpu=$MCPU"
    "-O3 -mcpu=$MCPU -flto"
    "-O3 -mcpu=$MCPU -flto -fomit-frame-pointer"
    "-O3 -mcpu=$MCPU -flto -funroll-loops"
    "-O3 -mcpu=$MCPU -flto -fvectorize -fslp-vectorize"
)

declare -a LABELS=(
    "No optimization"
    "Basic optimization"
    "Standard optimization"
    "Aggressive optimization"
    "Size optimization"
    "Aggressive size opt"
    "O3 + $MCPU tuning"
    "O3 + $MCPU + LTO"
    "O3 + $MCPU + LTO + no-frame-ptr"
    "O3 + $MCPU + LTO + unroll"
    "O3 + $MCPU + LTO + vectorize"
)

# Table header
printf "%-50s | %-12s | %-10s | %-12s | %s\n" \
    "Configuration" "Binary (KB)" "Train(ms)" "RMSE" "Notes" | tee -a "$RESULTS"
printf "%s\n" "$(printf '%.0s-' {1..120})" | tee -a "$RESULTS"

for i in "${!CONFIGS[@]}"; do
    FLAGS="${CONFIGS[$i]}"
    LABEL="${LABELS[$i]}"

    echo -n "  Compiling: $FLAGS ... " >&2

    # Compile
    if ! clang++ -std=c++17 $FLAGS -o gboost_test "$SOURCE" 2>/dev/null; then
        printf "%-50s | %-12s | %-10s | %-12s | %s\n" \
            "$FLAGS" "FAIL" "-" "-" "Compilation failed" | tee -a "$RESULTS"
        echo "FAILED" >&2
        continue
    fi

    # Binary size
    BIN_SIZE=$(ls -l gboost_test | awk '{print int($5/1024)}')

    # Run and capture output
    OUTPUT=$(./gboost_test 2>&1)

    # Extract metrics
    TRAIN_TIME=$(echo "$OUTPUT" | grep "TIMER.*Total training" | head -1 | \
                 grep -oE '[0-9]+\.[0-9]+' | head -1)
    RMSE=$(echo "$OUTPUT" | grep "Test RMSE" | head -1 | \
           grep -oE '[0-9]+\.[0-9]+' | head -1)

    printf "%-50s | %8s KB  | %8s ms | %12s | %s\n" \
        "$FLAGS" "$BIN_SIZE" "${TRAIN_TIME:-N/A}" "${RMSE:-N/A}" "$LABEL" | tee -a "$RESULTS"

    echo "done" >&2
done

# Cleanup
rm -f gboost_test

echo "" | tee -a "$RESULTS"
echo "Results saved to: $RESULTS" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

echo "============================================================" | tee -a "$RESULTS"
echo "  KEY OBSERVATIONS" | tee -a "$RESULTS"
echo "============================================================" | tee -a "$RESULTS"
echo "  Compare:" | tee -a "$RESULTS"
echo "    - O0 vs O3: raw optimization impact" | tee -a "$RESULTS"
echo "    - O3 vs O3+$MCPU: architecture-specific tuning" | tee -a "$RESULTS"
echo "    - O3+$MCPU vs O3+$MCPU+LTO: link-time optimization" | tee -a "$RESULTS"
echo "    - Os/Oz vs O3: speed vs size tradeoff" | tee -a "$RESULTS"
echo "    - All should produce identical RMSE (correctness)" | tee -a "$RESULTS"