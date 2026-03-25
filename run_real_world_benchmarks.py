#!/usr/bin/env python3
"""
run_real_world_benchmarks.py — Benchmark C++ GBDT vs XGBoost on real-world datasets.

Downloads datasets from sklearn/fetch, runs both implementations,
saves results and generates publication-quality figures.

Usage:
    source venv/bin/activate
    python run_real_world_benchmarks.py
"""

import time
import subprocess
import json
import os
import struct
import sys
import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_diabetes,
    fetch_covtype,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ============================================================================
#  Configuration
# ============================================================================
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
BINARY_DIR  = "/tmp/gbdt_bench_data"
CPP_SOURCE  = "gboost_neon.cpp"

N_TREES     = 50
MAX_DEPTH   = 6
MAX_BINS    = 256
LR          = 0.1
LAMBDA      = 1.0
SEED        = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(BINARY_DIR,  exist_ok=True)

# ============================================================================
#  Synthetic dataset generators (to match C++ Friedman #1)
# ============================================================================
def generate_friedman1(n_samples, n_features=20, noise=1.0, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_samples, n_features)).astype(np.float32)
    y = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20.0 * (X[:, 2] - 0.5) ** 2
         + 10.0 * X[:, 3]
         + 5.0  * X[:, 4]
         + rng.normal(0, noise, n_samples))
    return X, y.astype(np.float64)

# ============================================================================
#  Dataset loaders
# ============================================================================
def load_datasets():
    """Load multiple real-world + synthetic datasets, return as dict."""
    datasets = {}
    
    # 1. Friedman #1 (synthetic, matches C++ code)
    print("  Loading Friedman #1 (synthetic)...")
    X, y = generate_friedman1(50000, 20, noise=1.0, seed=42)
    X_test, y_test = generate_friedman1(10000, 20, noise=1.0, seed=123)
    datasets["Friedman #1"] = {
        "X_train": X, "y_train": y,
        "X_test": X_test, "y_test": y_test,
        "task": "regression",
        "n_features": 20, "description": "Synthetic (50K×20)"
    }

    # 2. California Housing
    print("  Loading California Housing...")
    cal = fetch_california_housing()
    X, y = cal.data.astype(np.float32), cal.target.astype(np.float64)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    datasets["California Housing"] = {
        "X_train": X_tr, "y_train": y_tr,
        "X_test": X_te, "y_test": y_te,
        "task": "regression",
        "n_features": X.shape[1], "description": f"Real ({X_tr.shape[0]}×{X.shape[1]})"
    }

    # 3. Diabetes
    print("  Loading Diabetes...")
    diab = load_diabetes()
    X, y = diab.data.astype(np.float32), diab.target.astype(np.float64)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    datasets["Diabetes"] = {
        "X_train": X_tr, "y_train": y_tr,
        "X_test": X_te, "y_test": y_te,
        "task": "regression",
        "n_features": X.shape[1], "description": f"Real ({X_tr.shape[0]}×{X.shape[1]})"
    }

    # 4. Covertype (large, real-world) — use subset for tractability
    print("  Loading Covertype (subset 50K)...")
    try:
        cov = fetch_covtype()
        X, y = cov.data.astype(np.float32), cov.target.astype(np.float64)
        # Subsample for tractability — use first 50K
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(X))[:60000]
        X, y = X[idx], y[idx]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
        datasets["Covertype"] = {
            "X_train": X_tr, "y_train": y_tr,
            "X_test": X_te, "y_test": y_te,
            "task": "regression",  # treat multi-class as regression for GBDT
            "n_features": X.shape[1], "description": f"Real ({X_tr.shape[0]}×{X.shape[1]})"
        }
    except Exception as e:
        print(f"    Skipping Covertype: {e}")

    # 5. Year Prediction MSD (fetch or synthetic larger)
    print("  Generating large synthetic dataset (100K×50)...")
    X, y = generate_friedman1(100000, 50, noise=1.0, seed=99)
    X_test, y_test = generate_friedman1(20000, 50, noise=1.0, seed=200)
    datasets["Friedman Large"] = {
        "X_train": X, "y_train": y,
        "X_test": X_test, "y_test": y_test,
        "task": "regression",
        "n_features": 50, "description": "Synthetic (100K×50)"
    }

    return datasets

# ============================================================================
#  Save dataset as binary for C++ consumption
# ============================================================================
def save_binary_dataset(name, X_train, y_train, X_test, y_test):
    """Save dataset as flat binary files for C++ to read."""
    prefix = os.path.join(BINARY_DIR, name.replace(" ", "_").lower())
    
    # Header: n_train, n_test, n_features (as uint64)
    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]
    
    with open(f"{prefix}_meta.bin", "wb") as f:
        f.write(struct.pack("QQQ", n_train, n_test, n_features))
    
    # Data files (float32 X, float64 y)
    X_train.astype(np.float32).tofile(f"{prefix}_X_train.bin")
    y_train.astype(np.float64).tofile(f"{prefix}_y_train.bin")
    X_test.astype(np.float32).tofile(f"{prefix}_X_test.bin")
    y_test.astype(np.float64).tofile(f"{prefix}_y_test.bin")
    
    return prefix

# ============================================================================
#  Run XGBoost benchmark
# ============================================================================
def run_xgboost(X_train, y_train, X_test, y_test, n_threads=1):
    """Run XGBoost with matched hyperparameters, return timing + RMSE."""
    model = xgb.XGBRegressor(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        learning_rate=LR,
        reg_lambda=LAMBDA,
        tree_method="hist",
        max_bin=MAX_BINS,
        n_jobs=n_threads,
        verbosity=0,
        random_state=SEED,
    )
    
    # Warmup
    model.fit(X_train[:100], y_train[:100])
    
    # Training
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_ms = (time.perf_counter() - t0) * 1000
    
    # Inference
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    infer_ms = (time.perf_counter() - t0) * 1000
    
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))
    
    return {
        "train_ms": round(train_ms, 2),
        "infer_ms": round(infer_ms, 2),
        "rmse": round(float(rmse), 6),
    }

# ============================================================================
#  Run C++ implementation via subprocess
# ============================================================================
def compile_cpp_runner():
    """Compile a special C++ binary that reads binary datasets."""
    runner_src = "/tmp/gbdt_bench_data/cpp_bench_runner.cpp"
    runner_bin = "/tmp/gbdt_bench_data/cpp_bench_runner"
    
    # Write a minimal C++ wrapper that reads binary data and runs our GBDT
    with open(runner_src, "w") as f:
        f.write("""
// Auto-generated benchmark runner — reads binary data files
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>

// Include the full GBDT implementation (it has everything we need)
// We'll just override main() approach by using sections directly

""")

    # Instead of including the full source, let's create a standalone runner
    # that reuses the DenseMatrix, GradientBoosting, etc. from gboost_neon.cpp
    
    runner_code = r'''
/*
 * cpp_bench_runner.cpp — Reads binary dataset and runs GBDT benchmark
 * Auto-generated by run_real_world_benchmarks.py
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#   include <arm_neon.h>
#   define HAS_NEON 1
#else
#   define HAS_NEON 0
#endif

// ---- Copy essential structures from gboost_neon.cpp ----

struct GBConfig {
    size_t   n_trees          = 100;
    size_t   max_depth        = 6;
    size_t   max_bins         = 256;
    size_t   min_samples_leaf = 5;
    double   learning_rate    = 0.1;
    double   lambda_l2        = 1.0;
    double   gamma            = 0.0;
    double   subsample        = 1.0;
    double   colsample        = 1.0;
    bool     use_simd         = true;
    bool     use_sparse       = false;
};

class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0_;
public:
    Timer() : t0_(Clock::now()) {}
    void reset() { t0_ = Clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
    }
};

struct DenseMatrix {
    std::vector<float> data;
    size_t n_rows = 0;
    size_t n_cols = 0;
    float  get(size_t r, size_t c) const { return data[r * n_cols + c]; }
    float& get(size_t r, size_t c)       { return data[r * n_cols + c]; }
    const float* row_ptr(size_t r) const { return &data[r * n_cols]; }
};

struct HistEntry {
    double   grad_sum  = 0.0;
    double   hess_sum  = 0.0;
    uint32_t count     = 0;
    uint32_t padding   = 0;
    void clear() { grad_sum = 0; hess_sum = 0; count = 0; }
};

struct Histogram {
    std::vector<HistEntry> entries;
    Histogram() = default;
    explicit Histogram(size_t n_bins) : entries(n_bins) {}
    void   clear() { for (auto& e : entries) e.clear(); }
    size_t size()  const { return entries.size(); }
};

struct BinMapper {
    std::vector<std::vector<float>> bin_edges;
    size_t max_bins;
    void fit(const DenseMatrix& X, size_t max_b) {
        max_bins = max_b;
        bin_edges.resize(X.n_cols);
        for (size_t f = 0; f < X.n_cols; ++f) {
            std::vector<float> vals(X.n_rows);
            for (size_t i = 0; i < X.n_rows; ++i)
                vals[i] = X.get(i, f);
            std::sort(vals.begin(), vals.end());
            vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
            size_t n_edges = std::min(max_bins - 1, vals.size());
            bin_edges[f].resize(n_edges);
            for (size_t b = 0; b < n_edges; ++b) {
                size_t idx = (b + 1) * vals.size() / (n_edges + 1);
                idx = std::min(idx, vals.size() - 1);
                bin_edges[f][b] = vals[idx];
            }
        }
    }
    uint8_t map_value(size_t feature, float value) const {
        const auto& edges = bin_edges[feature];
        auto it = std::lower_bound(edges.begin(), edges.end(), value);
        return static_cast<uint8_t>(it - edges.begin());
    }
};

struct ColumnBinnedData {
    std::vector<uint8_t> bins;
    size_t n_rows, n_cols;
    size_t max_bins;
    const uint8_t* col(size_t c) const { return &bins[c * n_rows]; }
};

ColumnBinnedData create_binned_data(const DenseMatrix& X, const BinMapper& mapper) {
    ColumnBinnedData bd;
    bd.n_rows = X.n_rows; bd.n_cols = X.n_cols; bd.max_bins = mapper.max_bins;
    bd.bins.resize(X.n_rows * X.n_cols);
    for (size_t c = 0; c < X.n_cols; ++c)
        for (size_t r = 0; r < X.n_rows; ++r)
            bd.bins[c * X.n_rows + r] = mapper.map_value(c, X.get(r, c));
    return bd;
}

// ---- Histogram construction (scalar + SIMD) ----

void build_histogram_scalar(
    const uint8_t* bin_col, const double* gradients, const double* hessians,
    const uint32_t* row_indices, size_t n_rows, Histogram& hist) {
    hist.clear();
    for (size_t i = 0; i < n_rows; ++i) {
        uint32_t row = row_indices[i];
        uint8_t bin = bin_col[row];
        hist.entries[bin].grad_sum += gradients[row];
        hist.entries[bin].hess_sum += hessians[row];
        hist.entries[bin].count += 1;
    }
}

#if HAS_NEON
static inline double hsum_f64x2(float64x2_t v) {
    return vgetq_lane_f64(v, 0) + vgetq_lane_f64(v, 1);
}

void build_histogram_simd(
    const uint8_t* bin_col, const double* gradients, const double* hessians,
    const uint32_t* row_indices, size_t n_rows, Histogram& hist) {
    hist.clear();
    if (n_rows < 16) {
        build_histogram_scalar(bin_col, gradients, hessians, row_indices, n_rows, hist);
        return;
    }
    size_t i = 0;
    const size_t n4 = n_rows & ~size_t(3);
    for (; i < n4; i += 4) {
        __builtin_prefetch(&row_indices[i + 16], 0, 1);
        uint32_t r0 = row_indices[i], r1 = row_indices[i+1];
        uint32_t r2 = row_indices[i+2], r3 = row_indices[i+3];
        uint8_t b0 = bin_col[r0], b1 = bin_col[r1];
        uint8_t b2 = bin_col[r2], b3 = bin_col[r3];
        float64x2_t g01 = {gradients[r0], gradients[r1]};
        float64x2_t g23 = {gradients[r2], gradients[r3]};
        float64x2_t h01 = {hessians[r0], hessians[r1]};
        float64x2_t h23 = {hessians[r2], hessians[r3]};
        if (b0 == b1 && b2 == b3 && b0 == b2) {
            float64x2_t gsum = vaddq_f64(g01, g23);
            float64x2_t hsum = vaddq_f64(h01, h23);
            hist.entries[b0].grad_sum += hsum_f64x2(gsum);
            hist.entries[b0].hess_sum += hsum_f64x2(hsum);
            hist.entries[b0].count += 4;
        } else {
            hist.entries[b0].grad_sum += gradients[r0];
            hist.entries[b0].hess_sum += hessians[r0];
            hist.entries[b0].count += 1;
            hist.entries[b1].grad_sum += gradients[r1];
            hist.entries[b1].hess_sum += hessians[r1];
            hist.entries[b1].count += 1;
            hist.entries[b2].grad_sum += gradients[r2];
            hist.entries[b2].hess_sum += hessians[r2];
            hist.entries[b2].count += 1;
            hist.entries[b3].grad_sum += gradients[r3];
            hist.entries[b3].hess_sum += hessians[r3];
            hist.entries[b3].count += 1;
        }
    }
    for (; i < n_rows; ++i) {
        uint32_t row = row_indices[i];
        uint8_t bin = bin_col[row];
        hist.entries[bin].grad_sum += gradients[row];
        hist.entries[bin].hess_sum += hessians[row];
        hist.entries[bin].count += 1;
    }
}

void compute_gradients_simd(
    const double* predictions, const double* targets,
    double* gradients, double* hessians, size_t n) {
    size_t i = 0;
    const float64x2_t ones = vdupq_n_f64(1.0);
    const size_t n4 = n & ~size_t(3);
    for (; i < n4; i += 4) {
        float64x2_t pred0 = vld1q_f64(&predictions[i]);
        float64x2_t tgt0  = vld1q_f64(&targets[i]);
        float64x2_t pred1 = vld1q_f64(&predictions[i + 2]);
        float64x2_t tgt1  = vld1q_f64(&targets[i + 2]);
        vst1q_f64(&gradients[i], vsubq_f64(pred0, tgt0));
        vst1q_f64(&gradients[i + 2], vsubq_f64(pred1, tgt1));
        vst1q_f64(&hessians[i], ones);
        vst1q_f64(&hessians[i + 2], ones);
    }
    for (; i < n; ++i) {
        gradients[i] = predictions[i] - targets[i];
        hessians[i] = 1.0;
    }
}

double compute_rmse_simd(const double* predictions, const double* targets, size_t n) {
    float64x2_t v_sse0 = vdupq_n_f64(0.0);
    float64x2_t v_sse1 = vdupq_n_f64(0.0);
    size_t i = 0;
    const size_t n4 = n & ~size_t(3);
    for (; i < n4; i += 4) {
        float64x2_t e0 = vsubq_f64(vld1q_f64(&predictions[i]), vld1q_f64(&targets[i]));
        float64x2_t e1 = vsubq_f64(vld1q_f64(&predictions[i+2]), vld1q_f64(&targets[i+2]));
        v_sse0 = vfmaq_f64(v_sse0, e0, e0);
        v_sse1 = vfmaq_f64(v_sse1, e1, e1);
    }
    double sse = hsum_f64x2(vaddq_f64(v_sse0, v_sse1));
    for (; i < n; ++i) { double e = predictions[i] - targets[i]; sse += e*e; }
    return std::sqrt(sse / (double)n);
}

#else
void build_histogram_simd(const uint8_t* bc, const double* g, const double* h,
    const uint32_t* ri, size_t n, Histogram& hist) {
    build_histogram_scalar(bc, g, h, ri, n, hist);
}
void compute_gradients_simd(const double* p, const double* t,
    double* g, double* h, size_t n) {
    for (size_t i = 0; i < n; ++i) { g[i] = p[i] - t[i]; h[i] = 1.0; }
}
double compute_rmse_simd(const double* p, const double* t, size_t n) {
    double sse = 0; for (size_t i = 0; i < n; ++i) { double e = p[i]-t[i]; sse += e*e; }
    return std::sqrt(sse / n);
}
#endif

// ---- Tree and boosting ----

struct TreeNode {
    int32_t feature_id = -1; uint8_t split_bin = 0;
    float split_value = 0.0f; double leaf_value = 0.0;
    int32_t left_child = -1, right_child = -1;
    bool is_leaf() const { return feature_id == -1; }
};

struct SplitInfo {
    int32_t feature = -1; uint8_t bin = 0;
    double gain = -std::numeric_limits<double>::infinity();
    double left_grad = 0, left_hess = 0, right_grad = 0, right_hess = 0;
    uint32_t left_count = 0, right_count = 0;
    bool valid() const { return gain > 0 && feature >= 0; }
};

class DecisionTree {
public:
    std::vector<TreeNode> nodes;
    size_t max_depth, max_bins, min_samples_leaf;
    double lambda, gamma;

    DecisionTree(const GBConfig& cfg)
        : max_depth(cfg.max_depth), max_bins(cfg.max_bins),
          min_samples_leaf(cfg.min_samples_leaf),
          lambda(cfg.lambda_l2), gamma(cfg.gamma) {
        nodes.reserve(1 << (max_depth + 1));
    }

    double compute_leaf_weight(double g, double h) const {
        return -g / (h + lambda);
    }

    double compute_gain(double lg, double lh, double rg, double rh) const {
        return 0.5 * ((lg*lg)/(lh+lambda) + (rg*rg)/(rh+lambda) -
               ((lg+rg)*(lg+rg))/(lh+rh+lambda)) - gamma;
    }

    SplitInfo find_best_split_feature(const Histogram& hist, int32_t fid,
                                       double tg, double th, uint32_t tc) const {
        SplitInfo best; double cg = 0, ch = 0; uint32_t cc = 0;
        for (size_t b = 0; b + 1 < hist.size(); ++b) {
            cg += hist.entries[b].grad_sum; ch += hist.entries[b].hess_sum;
            cc += hist.entries[b].count;
            if (cc < min_samples_leaf) continue;
            uint32_t rc = tc - cc; if (rc < min_samples_leaf) break;
            double rg = tg - cg, rh = th - ch;
            double gain = compute_gain(cg, ch, rg, rh);
            if (gain > best.gain) {
                best = {fid, (uint8_t)b, gain, cg, ch, rg, rh, cc, rc};
            }
        }
        return best;
    }

    int32_t build_node(const ColumnBinnedData& data, const BinMapper& mapper,
                       const double* gradients, const double* hessians,
                       std::vector<uint32_t>& rows, size_t depth, bool use_simd) {
        size_t n = rows.size();
        double tg = 0, th = 0;
        for (uint32_t r : rows) { tg += gradients[r]; th += hessians[r]; }
        if (depth >= max_depth || n <= min_samples_leaf) {
            int32_t nid = (int32_t)nodes.size(); nodes.push_back({});
            nodes.back().leaf_value = compute_leaf_weight(tg, th); return nid;
        }
        SplitInfo best; Histogram hist(data.max_bins);
        for (size_t f = 0; f < data.n_cols; ++f) {
            if (use_simd) build_histogram_simd(data.col(f), gradients, hessians, rows.data(), n, hist);
            else build_histogram_scalar(data.col(f), gradients, hessians, rows.data(), n, hist);
            SplitInfo sp = find_best_split_feature(hist, (int32_t)f, tg, th, (uint32_t)n);
            if (sp.gain > best.gain) best = sp;
        }
        if (!best.valid()) {
            int32_t nid = (int32_t)nodes.size(); nodes.push_back({});
            nodes.back().leaf_value = compute_leaf_weight(tg, th); return nid;
        }
        std::vector<uint32_t> left, right;
        left.reserve(best.left_count); right.reserve(best.right_count);
        const uint8_t* col = data.col(best.feature);
        for (uint32_t r : rows) {
            if (col[r] <= best.bin) left.push_back(r); else right.push_back(r);
        }
        int32_t nid = (int32_t)nodes.size(); nodes.push_back({});
        nodes[nid].feature_id = best.feature; nodes[nid].split_bin = best.bin;
        const auto& edges = mapper.bin_edges[best.feature];
        if (best.bin < edges.size()) nodes[nid].split_value = edges[best.bin];
        nodes[nid].left_child = build_node(data, mapper, gradients, hessians, left, depth+1, use_simd);
        nodes[nid].right_child = build_node(data, mapper, gradients, hessians, right, depth+1, use_simd);
        return nid;
    }

    double predict_one(const float* features) const {
        int32_t idx = 0;
        while (!nodes[idx].is_leaf()) {
            idx = (features[nodes[idx].feature_id] <= nodes[idx].split_value)
                ? nodes[idx].left_child : nodes[idx].right_child;
        }
        return nodes[idx].leaf_value;
    }

    double predict_one_binned(const ColumnBinnedData& data, uint32_t row) const {
        int32_t idx = 0;
        while (!nodes[idx].is_leaf()) {
            idx = (data.col(nodes[idx].feature_id)[row] <= nodes[idx].split_bin)
                ? nodes[idx].left_child : nodes[idx].right_child;
        }
        return nodes[idx].leaf_value;
    }
};

class GradientBoosting {
public:
    GBConfig config;
    std::vector<DecisionTree> trees;
    double base_score = 0.0;
    BinMapper bin_mapper;

    void fit(const DenseMatrix& X, const std::vector<double>& y) {
        size_t n = X.n_rows;
        bin_mapper.fit(X, config.max_bins);
        auto binned = create_binned_data(X, bin_mapper);
        base_score = 0.0;
        for (double yi : y) base_score += yi;
        base_score /= (double)n;
        std::vector<double> preds(n, base_score), grads(n), hess(n);
        std::vector<uint32_t> all_rows(n);
        std::iota(all_rows.begin(), all_rows.end(), 0u);
        for (size_t t = 0; t < config.n_trees; ++t) {
            if (config.use_simd)
                compute_gradients_simd(preds.data(), y.data(), grads.data(), hess.data(), n);
            else
                for (size_t i = 0; i < n; ++i) { grads[i] = preds[i] - y[i]; hess[i] = 1.0; }
            std::vector<uint32_t> rows = all_rows;
            DecisionTree tree(config);
            tree.build_node(binned, bin_mapper, grads.data(), hess.data(), rows, 0, config.use_simd);
            for (size_t i = 0; i < n; ++i)
                preds[i] += config.learning_rate * tree.predict_one_binned(binned, (uint32_t)i);
            trees.push_back(std::move(tree));
        }
    }

    std::vector<double> predict(const DenseMatrix& X) const {
        std::vector<double> preds(X.n_rows, base_score);
        for (const auto& tree : trees)
            for (size_t i = 0; i < X.n_rows; ++i)
                preds[i] += config.learning_rate * tree.predict_one(X.row_ptr(i));
        return preds;
    }

    double rmse(const std::vector<double>& preds, const std::vector<double>& truth) const {
        double sse = 0;
        for (size_t i = 0; i < preds.size(); ++i) { double e = preds[i]-truth[i]; sse += e*e; }
        return std::sqrt(sse / preds.size());
    }
};

// ---- BINARY I/O ----

bool load_binary_dataset(const std::string& prefix,
                         DenseMatrix& X_train, std::vector<double>& y_train,
                         DenseMatrix& X_test,  std::vector<double>& y_test) {
    // Read metadata
    std::ifstream meta(prefix + "_meta.bin", std::ios::binary);
    if (!meta) return false;
    uint64_t n_train, n_test, n_feat;
    meta.read((char*)&n_train, 8);
    meta.read((char*)&n_test, 8);
    meta.read((char*)&n_feat, 8);

    X_train.n_rows = n_train; X_train.n_cols = n_feat;
    X_train.data.resize(n_train * n_feat);
    X_test.n_rows = n_test; X_test.n_cols = n_feat;
    X_test.data.resize(n_test * n_feat);
    y_train.resize(n_train);
    y_test.resize(n_test);

    std::ifstream f1(prefix + "_X_train.bin", std::ios::binary);
    f1.read((char*)X_train.data.data(), n_train * n_feat * sizeof(float));
    std::ifstream f2(prefix + "_y_train.bin", std::ios::binary);
    f2.read((char*)y_train.data(), n_train * sizeof(double));
    std::ifstream f3(prefix + "_X_test.bin", std::ios::binary);
    f3.read((char*)X_test.data.data(), n_test * n_feat * sizeof(float));
    std::ifstream f4(prefix + "_y_test.bin", std::ios::binary);
    f4.read((char*)y_test.data(), n_test * sizeof(double));

    return true;
}

// ---- MAIN ----

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_prefix> [simd|scalar]\n";
        return 1;
    }
    std::string prefix = argv[1];
    bool use_simd = true;
    if (argc >= 3 && std::string(argv[2]) == "scalar") use_simd = false;

    DenseMatrix X_train, X_test;
    std::vector<double> y_train, y_test;

    if (!load_binary_dataset(prefix, X_train, y_train, X_test, y_test)) {
        std::cerr << "Failed to load dataset from: " << prefix << "\n";
        return 1;
    }

    GradientBoosting gb;
    gb.config.n_trees   = ''' + str(N_TREES) + r''';
    gb.config.max_depth = ''' + str(MAX_DEPTH) + r''';
    gb.config.max_bins  = ''' + str(MAX_BINS) + r''';
    gb.config.learning_rate = ''' + str(LR) + r''';
    gb.config.lambda_l2 = ''' + str(LAMBDA) + r''';
    gb.config.use_simd  = use_simd;

    // Training
    Timer t;
    gb.fit(X_train, y_train);
    double train_ms = t.elapsed_ms();

    // Inference
    Timer t2;
    auto preds = gb.predict(X_test);
    double infer_ms = t2.elapsed_ms();

    double rmse_val = gb.rmse(preds, y_test);

    // Output JSON-like result
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "{\"train_ms\": " << train_ms
              << ", \"infer_ms\": " << infer_ms
              << ", \"rmse\": " << std::setprecision(6) << rmse_val
              << ", \"mode\": \"" << (use_simd ? "neon" : "scalar") << "\"}" << std::endl;

    return 0;
}
'''
    
    with open(runner_src, "w") as f:
        f.write(runner_code)
    
    print("  Compiling C++ benchmark runner...")
    ret = subprocess.run(
        ["clang++", "-std=c++17", "-O3", "-mcpu=apple-m1",
         "-o", runner_bin, runner_src],
        capture_output=True, text=True
    )
    if ret.returncode != 0:
        print(f"  Compilation error: {ret.stderr}")
        return None
    
    print("  ✓ Compiled successfully")
    return runner_bin


def run_cpp_benchmark(runner_bin, prefix, mode="simd"):
    """Run C++ GBDT on a binary dataset, return parsed results."""
    try:
        ret = subprocess.run(
            [runner_bin, prefix, mode],
            capture_output=True, text=True, timeout=600
        )
        if ret.returncode != 0:
            print(f"  C++ error: {ret.stderr[:200]}")
            return None
        # Parse JSON output
        output = ret.stdout.strip()
        result = json.loads(output)
        return {
            "train_ms": round(result["train_ms"], 2),
            "infer_ms": round(result["infer_ms"], 2),
            "rmse": round(result["rmse"], 6),
        }
    except subprocess.TimeoutExpired:
        print("  C++ benchmark timed out (>600s)")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


# ============================================================================
#  Figure generation
# ============================================================================
def generate_all_figures(all_results):
    """Generate publication-quality comparison figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    dataset_names = list(all_results.keys())

    # ---- Figure 7: Training Time Comparison (C++ NEON vs XGBoost) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dataset_names))
    w = 0.25

    cpp_neon_times = [all_results[d].get("cpp_neon", {}).get("train_ms", 0) for d in dataset_names]
    cpp_scalar_times = [all_results[d].get("cpp_scalar", {}).get("train_ms", 0) for d in dataset_names]
    xgb_times = [all_results[d].get("xgboost_1t", {}).get("train_ms", 0) for d in dataset_names]

    bars1 = ax.bar(x - w, cpp_neon_times, w, label='C++ NEON', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, cpp_scalar_times, w, label='C++ Scalar', color='#3498db', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + w, xgb_times, w, label='XGBoost (1T)', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Training Time (ms)')
    ax.set_title('Training Time: C++ GBDT vs XGBoost\n(50 trees, depth 6, 256 bins, single-threaded)')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=15, ha='right')
    ax.legend()

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + max(max(cpp_neon_times), max(xgb_times))*0.01,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig7_multi_dataset_training.png')
    plt.savefig(f'{FIGURES_DIR}/fig7_multi_dataset_training.pdf')
    print("  ✓ fig7_multi_dataset_training")
    plt.close()

    # ---- Figure 8: RMSE Comparison ----
    fig, ax = plt.subplots(figsize=(10, 5))

    cpp_rmses = [all_results[d].get("cpp_neon", {}).get("rmse", 0) for d in dataset_names]
    xgb_rmses = [all_results[d].get("xgboost_1t", {}).get("rmse", 0) for d in dataset_names]

    bars1 = ax.bar(x - 0.15, cpp_rmses, 0.3, label='C++ NEON', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + 0.15, xgb_rmses, 0.3, label='XGBoost (1T)', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Test RMSE')
    ax.set_title('Accuracy Comparison: C++ GBDT vs XGBoost\n(lower = better)')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=15, ha='right')
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + max(max(cpp_rmses), max(xgb_rmses))*0.01,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig8_multi_dataset_rmse.png')
    plt.savefig(f'{FIGURES_DIR}/fig8_multi_dataset_rmse.pdf')
    print("  ✓ fig8_multi_dataset_rmse")
    plt.close()

    # ---- Figure 9: SIMD Speedup across datasets ----
    fig, ax = plt.subplots(figsize=(8, 5))

    speedups = []
    valid_names = []
    for d in dataset_names:
        neon = all_results[d].get("cpp_neon", {}).get("train_ms", 0)
        scalar = all_results[d].get("cpp_scalar", {}).get("train_ms", 0)
        if neon > 0 and scalar > 0:
            speedups.append(scalar / neon)
            valid_names.append(d)

    colors = ['#2ecc71' if s >= 1.0 else '#e74c3c' for s in speedups]
    bars = ax.bar(range(len(valid_names)), speedups, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No speedup')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Speedup (Scalar / NEON)')
    ax.set_title('NEON SIMD Speedup Across Datasets\n(>1.0 = NEON faster)')
    ax.set_xticks(range(len(valid_names)))
    ax.set_xticklabels(valid_names, rotation=15, ha='right')
    ax.legend()

    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{s:.2f}×', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig9_simd_speedup_datasets.png')
    plt.savefig(f'{FIGURES_DIR}/fig9_simd_speedup_datasets.pdf')
    print("  ✓ fig9_simd_speedup_datasets")
    plt.close()

    # ---- Figure 10: XGBoost vs C++ Performance Ratio ----
    fig, ax = plt.subplots(figsize=(8, 5))

    ratios = []
    ratio_names = []
    for d in dataset_names:
        cpp = all_results[d].get("cpp_neon", {}).get("train_ms", 0)
        xgb_t = all_results[d].get("xgboost_1t", {}).get("train_ms", 0)
        if cpp > 0 and xgb_t > 0:
            ratios.append(xgb_t / cpp)
            ratio_names.append(d)

    colors = ['#2ecc71' if r > 1.0 else '#e74c3c' for r in ratios]
    bars = ax.bar(range(len(ratio_names)), ratios, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Equal performance')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Ratio (XGBoost Time / C++ Time)')
    ax.set_title('C++ GBDT vs XGBoost: Training Time Ratio\n(>1.0 = C++ faster)')
    ax.set_xticks(range(len(ratio_names)))
    ax.set_xticklabels(ratio_names, rotation=15, ha='right')
    ax.legend()

    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r:.2f}×', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig10_xgboost_ratio.png')
    plt.savefig(f'{FIGURES_DIR}/fig10_xgboost_ratio.pdf')
    print("  ✓ fig10_xgboost_ratio")
    plt.close()



# ============================================================================
#  Main
# ============================================================================
def rmse_fn(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def main():
    print("=" * 70)
    print("  Multi-Dataset GBDT Benchmark: C++ (NEON/Scalar) vs XGBoost")
    print("=" * 70)

    # Step 1: Load datasets
    print("\n[1] Loading datasets...")
    datasets = load_datasets()
    print(f"    Loaded {len(datasets)} datasets\n")

    # Step 2: Compile C++ runner
    print("[2] Compiling C++ benchmark runner...")
    runner_bin = compile_cpp_runner()
    if runner_bin is None:
        print("  FATAL: Could not compile C++ runner. Continuing with XGBoost only.")

    # Step 3: Run benchmarks
    all_results = {}

    for name, ds in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {name} ({ds['description']})")
        print(f"  Train: {ds['X_train'].shape}, Test: {ds['X_test'].shape}")
        print(f"{'=' * 60}")

        result = {}

        # Save binary data for C++
        prefix = save_binary_dataset(name, ds["X_train"], ds["y_train"],
                                     ds["X_test"], ds["y_test"])

        # C++ NEON
        if runner_bin:
            print(f"\n  [C++ NEON]")
            cpp_neon = run_cpp_benchmark(runner_bin, prefix, "simd")
            if cpp_neon:
                result["cpp_neon"] = cpp_neon
                print(f"    Train: {cpp_neon['train_ms']:.1f} ms, "
                      f"RMSE: {cpp_neon['rmse']:.6f}")

        # C++ Scalar
        if runner_bin:
            print(f"  [C++ Scalar]")
            cpp_scalar = run_cpp_benchmark(runner_bin, prefix, "scalar")
            if cpp_scalar:
                result["cpp_scalar"] = cpp_scalar
                print(f"    Train: {cpp_scalar['train_ms']:.1f} ms, "
                      f"RMSE: {cpp_scalar['rmse']:.6f}")

        # XGBoost 1-thread
        print(f"  [XGBoost 1-thread]")
        xgb_1t = run_xgboost(ds["X_train"], ds["y_train"],
                             ds["X_test"], ds["y_test"], n_threads=1)
        result["xgboost_1t"] = xgb_1t
        print(f"    Train: {xgb_1t['train_ms']:.1f} ms, RMSE: {xgb_1t['rmse']:.6f}")

        # XGBoost 4-thread
        print(f"  [XGBoost 4-thread]")
        xgb_4t = run_xgboost(ds["X_train"], ds["y_train"],
                             ds["X_test"], ds["y_test"], n_threads=4)
        result["xgboost_4t"] = xgb_4t
        print(f"    Train: {xgb_4t['train_ms']:.1f} ms, RMSE: {xgb_4t['rmse']:.6f}")

        all_results[name] = result

    # Step 4: Print summary table
    print(f"\n\n{'=' * 100}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 100}")
    print(f"\n  {'Dataset':<20} {'C++ NEON':>15} {'C++ Scalar':>15} "
          f"{'XGB 1T':>15} {'XGB 4T':>15}  {'NEON RMSE':>12} {'XGB RMSE':>12}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15}  {'-'*12} {'-'*12}")

    for name, result in all_results.items():
        cpp_n = result.get("cpp_neon", {}).get("train_ms", "-")
        cpp_s = result.get("cpp_scalar", {}).get("train_ms", "-")
        xgb1 = result.get("xgboost_1t", {}).get("train_ms", "-")
        xgb4 = result.get("xgboost_4t", {}).get("train_ms", "-")
        rmse_n = result.get("cpp_neon", {}).get("rmse", "-")
        rmse_x = result.get("xgboost_1t", {}).get("rmse", "-")

        cpp_n_str = f"{cpp_n:.1f} ms" if isinstance(cpp_n, (int, float)) else cpp_n
        cpp_s_str = f"{cpp_s:.1f} ms" if isinstance(cpp_s, (int, float)) else cpp_s
        xgb1_str = f"{xgb1:.1f} ms" if isinstance(xgb1, (int, float)) else xgb1
        xgb4_str = f"{xgb4:.1f} ms" if isinstance(xgb4, (int, float)) else xgb4
        rmse_n_str = f"{rmse_n:.6f}" if isinstance(rmse_n, (int, float)) else rmse_n
        rmse_x_str = f"{rmse_x:.6f}" if isinstance(rmse_x, (int, float)) else rmse_x

        print(f"  {name:<20} {cpp_n_str:>15} {cpp_s_str:>15} "
              f"{xgb1_str:>15} {xgb4_str:>15}  {rmse_n_str:>12} {rmse_x_str:>12}")

    # Step 5: Save results
    results_file = os.path.join(RESULTS_DIR, "multi_dataset_results.json")
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_results = {}
    for name, result in all_results.items():
        serializable_results[name] = {}
        for method, data in result.items():
            serializable_results[name][method] = {
                k: convert_to_serializable(v) for k, v in data.items()
            }

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    # Step 6: Generate figures
    print("\n[6] Generating figures...")
    generate_all_figures(all_results)

    print(f"\n{'=' * 70}")
    print(f"  All benchmarks complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
