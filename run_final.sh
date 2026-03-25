#!/bin/bash
# run_final.sh — Master script: reproduce ALL results for the paper
# Usage: chmod +x run_final.sh && ./run_final.sh
#
# Prerequisites:
#   - macOS with Apple Silicon (M1/M2/M3)
#   - Xcode Command Line Tools (clang++)
#   - Python 3.10+ with pip
#
# This script will:
#   1. Compile and run the core C++ GBDT experiments
#   2. Run compiler flag ablation study
#   3. Run profiling
#   4. Set up Python virtualenv and run multi-dataset benchmarks
#   5. Run advanced benchmarks (HW counters, MT scaling, XGBoost breakdown)
#   6. Run logistic loss benchmark
#   7. Generate all figures

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-detect Apple Silicon model for -mcpu flag
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
echo "  GBDT ARM NEON Benchmark Suite"
echo "  $(date)"
echo "  CPU: $CHIP  (using -mcpu=$MCPU)"
echo "  $(clang++ --version | head -1)"
echo "============================================================"
echo ""

# ---- Step 1: Core C++ experiments ----
echo "[1/7] Compiling and running core C++ GBDT..."
clang++ -std=c++17 -O3 -mcpu=$MCPU -o gboost gboost_neon.cpp
./gboost 2>&1 | tee full_output.txt
echo ""

# ---- Step 2: Compiler ablation ----
echo "[2/7] Running compiler flag ablation..."
chmod +x run_ablation.sh
./run_ablation.sh
echo ""

# ---- Step 3: Profiling ----
echo "[3/7] Running profiling..."
chmod +x run_profile.sh
./run_profile.sh
echo ""

# ---- Step 4: Python environment + multi-dataset benchmarks ----
echo "[4/7] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --quiet xgboost numpy scikit-learn matplotlib

echo ""
echo "[5/7] Running multi-dataset benchmarks (C++ vs XGBoost)..."
python3 run_real_world_benchmarks.py

echo ""
echo "[6/7] Running advanced benchmarks (HW counters, MT, XGBoost breakdown)..."
python3 run_advanced_benchmarks.py

# ---- Step 5: Logistic loss benchmark ----
echo ""
echo "[7/7] Running logistic loss benchmark..."
clang++ -std=c++17 -O3 -mcpu=$MCPU -o /tmp/logistic_loss_bench /tmp/logistic_loss_bench.cpp 2>/dev/null || \
clang++ -std=c++17 -O3 -mcpu=$MCPU -o /tmp/logistic_loss_bench logistic_loss_bench.cpp 2>/dev/null || \
echo "  (logistic loss bench source not found, skipping)"

if [ -f /tmp/logistic_loss_bench ]; then
    /tmp/logistic_loss_bench | tee results/logistic_loss_results.json
fi

deactivate 2>/dev/null || true

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo ""
echo "  Output files:"
echo "    full_output.txt                    — core C++ output"
echo "    ablation_results.txt               — compiler flag study"
echo "    results/multi_dataset_results.json — 5-dataset comparison"
echo "    results/advanced_results.json      — HW counters & MT"
echo "    results/logistic_loss_results.json — loss function analysis"
echo "    figures/*.pdf                      — publication figures"
echo "============================================================"