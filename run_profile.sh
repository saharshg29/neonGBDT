#!/bin/bash
# run_profile.sh — Profile the GBDT implementation
# Compiles with debug symbols and runs built-in timers
# Usage: chmod +x run_profile.sh && ./run_profile.sh

set -euo pipefail

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

echo "============================================================"
echo "  PROFILING GRADIENT BOOSTING"
echo "  CPU: $CHIP  (using -mcpu=$MCPU)"
echo "============================================================"

# Compile with debug symbols + optimization
echo "[1] Compiling with debug symbols..."
clang++ -std=c++17 -O3 -mcpu=$MCPU -g -o gboost_profile gboost_neon.cpp

# Get binary size info
echo "[2] Binary info:"
ls -lh gboost_profile
echo ""

# Run with timing
echo "[3] Running with built-in timers..."
echo "------------------------------------------------------------"
./gboost_profile 2>&1 | tee full_output.txt
echo "------------------------------------------------------------"
echo ""

# Extract timing summary
echo "[4] Timing Summary:"
grep "TIMER" full_output.txt | sort -t: -k2 -n -r || echo "  (no TIMER lines found)"
echo ""

# Memory usage (macOS)
echo "[5] Peak memory usage:"
/usr/bin/time -l ./gboost_profile 2>&1 | grep "maximum resident" || echo "  (could not measure)"
echo ""

# Save profile summary
cat > profile_summary.txt <<EOF
============================================================
  PROFILE SUMMARY — $(date)
  CPU: $CHIP
============================================================

Timing breakdown:
$(grep "TIMER" full_output.txt 2>/dev/null | sort -t: -k2 -n -r || echo "  N/A")

Cache hierarchy (from sysctl):
  L1d: $(sysctl -n hw.l1dcachesize 2>/dev/null | awk '{printf "%.0f KB", $1/1024}' || echo "N/A")
  L2:  $(sysctl -n hw.l2cachesize 2>/dev/null | awk '{printf "%.1f MB", $1/1048576}' || echo "N/A")
  Cache line: $(sysctl -n hw.cachelinesize 2>/dev/null || echo "N/A") bytes
  Cores: $(sysctl -n hw.ncpu 2>/dev/null || echo "N/A") ($(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "?")P + $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo "?")E)
EOF

echo ""
echo "============================================================"
echo "  Output files:"
echo "    full_output.txt       — complete program output"
echo "    profile_summary.txt   — timing + hardware summary"
echo "============================================================"