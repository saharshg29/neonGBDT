#!/usr/bin/env python3
"""
run_advanced_benchmarks.py — Hardware counters, multi-threading, XGBoost kernel breakdown.

Produces:
  - Cache/branch/IPC analysis via C++ micro-benchmarks
  - Thread scaling curves (1,2,4,8 threads) with SIMD interaction
  - XGBoost kernel-level timing breakdown
  - New figures & JSON results for paper

Platform: Apple M3, 4P+4E cores, L1d=64KB, L2=4MB, 128B cache line, 16GB
"""

import time, json, os, subprocess, struct, sys
import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_covtype
from sklearn.model_selection import train_test_split

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SEED = 42

# ============================================================================
#  PART 1: Hardware Counter Micro-Benchmarks (C++)
# ============================================================================

HW_COUNTER_SRC = r'''
/*
 * hw_counters_bench.cpp — Cache, branch, IPC micro-benchmarks for GBDT kernels
 * Apple M3: L1d=64KB, L2=4MB, cache line=128B
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <sys/resource.h>
#include <mach/mach.h>
#include <mach/thread_info.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#define HAS_NEON 1
#endif

using Clock = std::chrono::high_resolution_clock;

static inline uint64_t rdtsc_approx() {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t rdtsc_freq() {
    uint64_t val;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

struct RUsageSnap {
    struct rusage ru;
    void capture() { getrusage(RUSAGE_SELF, &ru); }
    long page_faults() const { return ru.ru_minflt + ru.ru_majflt; }
    long major_faults() const { return ru.ru_majflt; }
    long vol_ctx() const { return ru.ru_nvcsw; }
    long invol_ctx() const { return ru.ru_nivcsw; }
};

// ============================================================================
//  Histogram entry & construction (same as paper)
// ============================================================================
struct HistEntry {
    double grad_sum = 0.0, hess_sum = 0.0;
    uint32_t count = 0, pad = 0;
    void clear() { grad_sum = 0; hess_sum = 0; count = 0; }
};

void hist_scalar(const uint8_t* bins, const double* g, const double* h,
                 const uint32_t* rows, size_t n, HistEntry* hist, size_t nbins) {
    for (size_t i = 0; i < nbins; ++i) hist[i].clear();
    for (size_t i = 0; i < n; ++i) {
        uint32_t r = rows[i];
        uint8_t b = bins[r];
        hist[b].grad_sum += g[r];
        hist[b].hess_sum += h[r];
        hist[b].count += 1;
    }
}

#if HAS_NEON
static inline double hsum_f64x2(float64x2_t v) {
    return vgetq_lane_f64(v, 0) + vgetq_lane_f64(v, 1);
}

void hist_neon(const uint8_t* bins, const double* g, const double* h,
               const uint32_t* rows, size_t n, HistEntry* hist, size_t nbins) {
    for (size_t i = 0; i < nbins; ++i) hist[i].clear();
    size_t i = 0;
    const size_t n4 = n & ~size_t(3);
    for (; i < n4; i += 4) {
        __builtin_prefetch(&rows[i + 16], 0, 1);
        uint32_t r0=rows[i], r1=rows[i+1], r2=rows[i+2], r3=rows[i+3];
        uint8_t b0=bins[r0], b1=bins[r1], b2=bins[r2], b3=bins[r3];
        if (b0==b1 && b2==b3 && b0==b2) {
            float64x2_t gs = vaddq_f64(
                float64x2_t{g[r0],g[r1]}, float64x2_t{g[r2],g[r3]});
            float64x2_t hs = vaddq_f64(
                float64x2_t{h[r0],h[r1]}, float64x2_t{h[r2],h[r3]});
            hist[b0].grad_sum += hsum_f64x2(gs);
            hist[b0].hess_sum += hsum_f64x2(hs);
            hist[b0].count += 4;
        } else {
            hist[b0].grad_sum+=g[r0]; hist[b0].hess_sum+=h[r0]; hist[b0].count++;
            hist[b1].grad_sum+=g[r1]; hist[b1].hess_sum+=h[r1]; hist[b1].count++;
            hist[b2].grad_sum+=g[r2]; hist[b2].hess_sum+=h[r2]; hist[b2].count++;
            hist[b3].grad_sum+=g[r3]; hist[b3].hess_sum+=h[r3]; hist[b3].count++;
        }
    }
    for (; i < n; ++i) {
        uint32_t r=rows[i]; uint8_t b=bins[r];
        hist[b].grad_sum+=g[r]; hist[b].hess_sum+=h[r]; hist[b].count++;
    }
}
#endif

// ============================================================================
//  BENCHMARK 1: Cache working-set analysis
//  Vary data size to show L1/L2/DRAM transition in histogram construction
// ============================================================================
void bench_cache_working_set() {
    std::cout << "{\"bench\": \"cache_working_set\", \"data\": [";
    
    const size_t NBINS = 256;
    std::vector<HistEntry> hist(NBINS);
    std::mt19937 rng(42);
    
    // Sizes chosen to span L1d (64KB) → L2 (4MB) → DRAM
    // Each sample needs: 1B (bin) + 8B (grad) + 8B (hess) + 4B (row_idx) = 21 bytes
    // L1d fits ~3000 samples, L2 fits ~200K samples
    size_t sizes[] = {512, 1024, 2048, 4096, 8192, 16384, 32768,
                      65536, 131072, 262144, 524288, 1048576};
    bool first = true;
    
    for (size_t N : sizes) {
        std::vector<uint8_t> bins(N);
        std::vector<double> grads(N), hess(N);
        std::vector<uint32_t> rows(N);
        
        for (size_t i = 0; i < N; ++i) {
            bins[i] = rng() % NBINS;
            grads[i] = 0.1 * (rng() % 100 - 50);
            hess[i] = 1.0;
            rows[i] = (uint32_t)i;
        }
        
        // Warmup
        hist_scalar(bins.data(), grads.data(), hess.data(), rows.data(), N, hist.data(), NBINS);
        
        // Measure scalar
        const int ITERS = std::max(1, (int)(50000000 / N));
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it)
            hist_scalar(bins.data(), grads.data(), hess.data(), rows.data(), N, hist.data(), NBINS);
        double scalar_ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / ITERS;
        
#if HAS_NEON
        // Measure NEON
        t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it)
            hist_neon(bins.data(), grads.data(), hess.data(), rows.data(), N, hist.data(), NBINS);
        double neon_ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / ITERS;
#else
        double neon_ns = scalar_ns;
#endif
        
        double working_set_kb = N * 21.0 / 1024.0;
        double scalar_ns_per_elem = scalar_ns / N;
        double neon_ns_per_elem = neon_ns / N;
        
        if (!first) std::cout << ", ";
        first = false;
        std::cout << std::fixed << std::setprecision(2)
                  << "{\"n\": " << N
                  << ", \"ws_kb\": " << working_set_kb
                  << ", \"scalar_ns_elem\": " << scalar_ns_per_elem
                  << ", \"neon_ns_elem\": " << neon_ns_per_elem
                  << ", \"speedup\": " << scalar_ns_per_elem / neon_ns_per_elem
                  << "}";
    }
    std::cout << "]}" << std::endl;
}

// ============================================================================
//  BENCHMARK 2: Branch misprediction — sorted vs random bin access
// ============================================================================
void bench_branch_prediction() {
    std::cout << "{\"bench\": \"branch_prediction\", \"data\": [";
    
    const size_t N = 100000;
    const size_t NBINS = 256;
    std::vector<HistEntry> hist(NBINS);
    std::mt19937 rng(42);
    
    std::vector<double> grads(N), hess(N);
    std::vector<uint32_t> rows(N);
    for (size_t i = 0; i < N; ++i) {
        grads[i] = 0.1; hess[i] = 1.0; rows[i] = (uint32_t)i;
    }
    
    // Test different bin distributions (affects branch prediction in NEON path)
    struct TestCase {
        const char* name;
        int n_unique_bins;  // how many unique bins — more = more branch mispredict
    };
    TestCase cases[] = {
        {"1_bin", 1}, {"2_bins", 2}, {"4_bins", 4}, {"16_bins", 16},
        {"64_bins", 64}, {"128_bins", 128}, {"256_bins", 256}
    };
    
    bool first = true;
    for (auto& tc : cases) {
        std::vector<uint8_t> bins(N);
        for (size_t i = 0; i < N; ++i)
            bins[i] = rng() % tc.n_unique_bins;
        
        // Count collisions (same bin in groups of 4)
        int collisions = 0, total_groups = 0;
        for (size_t i = 0; i + 3 < N; i += 4) {
            total_groups++;
            if (bins[i]==bins[i+1] && bins[i+2]==bins[i+3] && bins[i]==bins[i+2])
                collisions++;
        }
        double collision_rate = (double)collisions / total_groups;
        
        const int ITERS = 200;
        
        // Warmup
        hist_scalar(bins.data(), grads.data(), hess.data(), rows.data(), N, hist.data(), NBINS);
        
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it)
            hist_scalar(bins.data(), grads.data(), hess.data(), rows.data(), N, hist.data(), NBINS);
        double scalar_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / ITERS;
        
#if HAS_NEON
        t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it)
            hist_neon(bins.data(), grads.data(), hess.data(), rows.data(), N, hist.data(), NBINS);
        double neon_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / ITERS;
#else
        double neon_us = scalar_us;
#endif
        
        if (!first) std::cout << ", ";
        first = false;
        std::cout << std::fixed << std::setprecision(3)
                  << "{\"bins\": " << tc.n_unique_bins
                  << ", \"collision_rate\": " << std::setprecision(4) << collision_rate
                  << ", \"scalar_us\": " << std::setprecision(1) << scalar_us
                  << ", \"neon_us\": " << neon_us
                  << ", \"speedup\": " << std::setprecision(3) << scalar_us / neon_us
                  << ", \"name\": \"" << tc.name << "\"}";
    }
    std::cout << "]}" << std::endl;
}

// ============================================================================
//  BENCHMARK 3: IPC estimation — cycles per element
// ============================================================================
void bench_ipc() {
    std::cout << "{\"bench\": \"ipc_estimation\", \"data\": [";
    
    const size_t N = 200000;
    const size_t NBINS = 256;
    std::vector<HistEntry> hist(NBINS);
    std::mt19937 rng(42);
    
    std::vector<uint8_t> bins(N);
    std::vector<double> grads(N), hess(N);
    std::vector<uint32_t> rows(N);
    for (size_t i = 0; i < N; ++i) {
        bins[i] = rng() % NBINS;
        grads[i] = 0.1; hess[i] = 1.0; rows[i] = (uint32_t)i;
    }
    
    uint64_t freq = rdtsc_freq();
    
    struct RUsageSnap before, after;
    
    // Scalar
    hist_scalar(bins.data(), grads.data(), hess.data(), rows.data(), N, hist.data(), NBINS);
    before.capture();
    uint64_t c0 = rdtsc_approx();
    const int ITERS = 500;
    for (int it = 0; it < ITERS; ++it)
        hist_scalar(bins.data(), grads.data(), hess.data(), rows.data(), N, hist.data(), NBINS);
    uint64_t c1 = rdtsc_approx();
    after.capture();
    
    double scalar_cycles_total = (double)(c1 - c0);
    double scalar_cycles_per_elem = scalar_cycles_total / ((double)N * ITERS);
    long scalar_faults = after.page_faults() - before.page_faults();
    
#if HAS_NEON
    // NEON
    hist_neon(bins.data(), grads.data(), hess.data(), rows.data(), N, hist.data(), NBINS);
    before.capture();
    c0 = rdtsc_approx();
    for (int it = 0; it < ITERS; ++it)
        hist_neon(bins.data(), grads.data(), hess.data(), rows.data(), N, hist.data(), NBINS);
    c1 = rdtsc_approx();
    after.capture();
    
    double neon_cycles_total = (double)(c1 - c0);
    double neon_cycles_per_elem = neon_cycles_total / ((double)N * ITERS);
    long neon_faults = after.page_faults() - before.page_faults();
#else
    double neon_cycles_per_elem = scalar_cycles_per_elem;
    long neon_faults = scalar_faults;
#endif

    // Gradient computation (embarrassingly parallel — should have high IPC)
    std::vector<double> preds(N, 0.5), targets(N, 0.3), grad_out(N), hess_out(N);
    
    c0 = rdtsc_approx();
    for (int it = 0; it < ITERS; ++it) {
        for (size_t i = 0; i < N; ++i) {
            grad_out[i] = preds[i] - targets[i];
            hess_out[i] = 1.0;
        }
    }
    c1 = rdtsc_approx();
    double grad_scalar_cpe = (double)(c1-c0) / ((double)N * ITERS);

#if HAS_NEON
    const float64x2_t ones = vdupq_n_f64(1.0);
    c0 = rdtsc_approx();
    for (int it = 0; it < ITERS; ++it) {
        size_t i = 0;
        const size_t n4 = N & ~size_t(3);
        for (; i < n4; i += 4) {
            float64x2_t p0 = vld1q_f64(&preds[i]);
            float64x2_t t0 = vld1q_f64(&targets[i]);
            float64x2_t p1 = vld1q_f64(&preds[i+2]);
            float64x2_t t1 = vld1q_f64(&targets[i+2]);
            vst1q_f64(&grad_out[i], vsubq_f64(p0, t0));
            vst1q_f64(&grad_out[i+2], vsubq_f64(p1, t1));
            vst1q_f64(&hess_out[i], ones);
            vst1q_f64(&hess_out[i+2], ones);
        }
        for (; i < N; ++i) { grad_out[i] = preds[i]-targets[i]; hess_out[i] = 1.0; }
    }
    c1 = rdtsc_approx();
    double grad_neon_cpe = (double)(c1-c0) / ((double)N * ITERS);
#else
    double grad_neon_cpe = grad_scalar_cpe;
#endif

    std::cout << std::fixed << std::setprecision(2)
              << "{\"kernel\": \"histogram\", \"scalar_cpe\": " << scalar_cycles_per_elem
              << ", \"neon_cpe\": " << neon_cycles_per_elem
              << ", \"scalar_faults\": " << scalar_faults
              << ", \"neon_faults\": " << neon_faults
              << ", \"timer_freq_mhz\": " << freq / 1000000.0 << "}, "
              << "{\"kernel\": \"gradient\", \"scalar_cpe\": " << grad_scalar_cpe
              << ", \"neon_cpe\": " << grad_neon_cpe
              << ", \"scalar_faults\": 0, \"neon_faults\": 0"
              << ", \"timer_freq_mhz\": " << freq / 1000000.0 << "}";
    
    std::cout << "]}" << std::endl;
}

// ============================================================================
//  BENCHMARK 4: Memory access pattern — sequential vs random
// ============================================================================
void bench_access_pattern() {
    std::cout << "{\"bench\": \"access_pattern\", \"data\": [";
    
    const size_t N = 200000;
    const size_t NBINS = 256;
    std::vector<HistEntry> hist(NBINS);
    std::mt19937 rng(42);
    
    std::vector<uint8_t> bins(N);
    std::vector<double> grads(N), hess(N);
    for (size_t i = 0; i < N; ++i) {
        bins[i] = rng() % NBINS; grads[i] = 0.1; hess[i] = 1.0;
    }
    
    // Sequential rows vs randomly permuted rows
    std::vector<uint32_t> seq_rows(N), rand_rows(N);
    std::iota(seq_rows.begin(), seq_rows.end(), 0u);
    rand_rows = seq_rows;
    std::shuffle(rand_rows.begin(), rand_rows.end(), rng);
    
    const int ITERS = 200;
    
    // Sequential scalar
    hist_scalar(bins.data(), grads.data(), hess.data(), seq_rows.data(), N, hist.data(), NBINS);
    auto t0 = Clock::now();
    for (int it = 0; it < ITERS; ++it)
        hist_scalar(bins.data(), grads.data(), hess.data(), seq_rows.data(), N, hist.data(), NBINS);
    double seq_scalar_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / ITERS;
    
    // Random scalar
    t0 = Clock::now();
    for (int it = 0; it < ITERS; ++it)
        hist_scalar(bins.data(), grads.data(), hess.data(), rand_rows.data(), N, hist.data(), NBINS);
    double rand_scalar_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / ITERS;

#if HAS_NEON
    // Sequential NEON
    t0 = Clock::now();
    for (int it = 0; it < ITERS; ++it)
        hist_neon(bins.data(), grads.data(), hess.data(), seq_rows.data(), N, hist.data(), NBINS);
    double seq_neon_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / ITERS;
    
    // Random NEON
    t0 = Clock::now();
    for (int it = 0; it < ITERS; ++it)
        hist_neon(bins.data(), grads.data(), hess.data(), rand_rows.data(), N, hist.data(), NBINS);
    double rand_neon_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / ITERS;
#else
    double seq_neon_us = seq_scalar_us, rand_neon_us = rand_scalar_us;
#endif

    std::cout << std::fixed << std::setprecision(1)
              << "{\"pattern\": \"sequential\", \"scalar_us\": " << seq_scalar_us
              << ", \"neon_us\": " << seq_neon_us << "}, "
              << "{\"pattern\": \"random\", \"scalar_us\": " << rand_scalar_us
              << ", \"neon_us\": " << rand_neon_us << "}";
    
    std::cout << "]}" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <cache|branch|ipc|access>\n";
        return 1;
    }
    std::string bench = argv[1];
    if (bench == "cache") bench_cache_working_set();
    else if (bench == "branch") bench_branch_prediction();
    else if (bench == "ipc") bench_ipc();
    else if (bench == "access") bench_access_pattern();
    else { std::cerr << "Unknown: " << bench << "\n"; return 1; }
    return 0;
}
'''

# ============================================================================
#  PART 2: Multi-Threading C++ Benchmark
# ============================================================================

MT_BENCH_SRC = r'''
/*
 * mt_bench.cpp — Multi-threaded GBDT training with thread scaling analysis
 * Tests 1,2,4,8 threads with SIMD interaction
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#define HAS_NEON 1
#endif

using Clock = std::chrono::high_resolution_clock;

// ---- Structures ----
struct HistEntry {
    double grad_sum=0, hess_sum=0; uint32_t count=0, pad=0;
    void clear() { grad_sum=0; hess_sum=0; count=0; }
};

struct BinMapper {
    std::vector<std::vector<float>> bin_edges;
    size_t max_bins;
    void fit(const float* data, size_t n_rows, size_t n_cols, size_t max_b) {
        max_bins = max_b;
        bin_edges.resize(n_cols);
        for (size_t f = 0; f < n_cols; ++f) {
            std::vector<float> vals(n_rows);
            for (size_t i = 0; i < n_rows; ++i) vals[i] = data[i * n_cols + f];
            std::sort(vals.begin(), vals.end());
            vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
            size_t ne = std::min(max_bins - 1, vals.size());
            bin_edges[f].resize(ne);
            for (size_t b = 0; b < ne; ++b) {
                size_t idx = std::min((b+1)*vals.size()/(ne+1), vals.size()-1);
                bin_edges[f][b] = vals[idx];
            }
        }
    }
    uint8_t map(size_t f, float v) const {
        auto& e = bin_edges[f];
        return (uint8_t)(std::lower_bound(e.begin(), e.end(), v) - e.begin());
    }
};

// ---- Histogram construction ----
void hist_scalar(const uint8_t* bins, const double* g, const double* h,
                 const uint32_t* rows, size_t n, HistEntry* hist, size_t nb) {
    for (size_t i = 0; i < nb; ++i) hist[i].clear();
    for (size_t i = 0; i < n; ++i) {
        uint32_t r = rows[i]; uint8_t b = bins[r];
        hist[b].grad_sum += g[r]; hist[b].hess_sum += h[r]; hist[b].count++;
    }
}

#if HAS_NEON
static inline double hsum_f64x2(float64x2_t v) {
    return vgetq_lane_f64(v,0) + vgetq_lane_f64(v,1);
}
void hist_neon(const uint8_t* bins, const double* g, const double* h,
               const uint32_t* rows, size_t n, HistEntry* hist, size_t nb) {
    for (size_t i = 0; i < nb; ++i) hist[i].clear();
    size_t i = 0; const size_t n4 = n & ~size_t(3);
    for (; i < n4; i += 4) {
        uint32_t r0=rows[i],r1=rows[i+1],r2=rows[i+2],r3=rows[i+3];
        uint8_t b0=bins[r0],b1=bins[r1],b2=bins[r2],b3=bins[r3];
        if (b0==b1&&b2==b3&&b0==b2) {
            float64x2_t gs=vaddq_f64(float64x2_t{g[r0],g[r1]},float64x2_t{g[r2],g[r3]});
            float64x2_t hs=vaddq_f64(float64x2_t{h[r0],h[r1]},float64x2_t{h[r2],h[r3]});
            hist[b0].grad_sum+=hsum_f64x2(gs); hist[b0].hess_sum+=hsum_f64x2(hs); hist[b0].count+=4;
        } else {
            hist[b0].grad_sum+=g[r0];hist[b0].hess_sum+=h[r0];hist[b0].count++;
            hist[b1].grad_sum+=g[r1];hist[b1].hess_sum+=h[r1];hist[b1].count++;
            hist[b2].grad_sum+=g[r2];hist[b2].hess_sum+=h[r2];hist[b2].count++;
            hist[b3].grad_sum+=g[r3];hist[b3].hess_sum+=h[r3];hist[b3].count++;
        }
    }
    for (; i < n; ++i) {
        uint32_t r=rows[i]; uint8_t b=bins[r];
        hist[b].grad_sum+=g[r]; hist[b].hess_sum+=h[r]; hist[b].count++;
    }
}
#else
void hist_neon(const uint8_t* b, const double* g, const double* h,
               const uint32_t* r, size_t n, HistEntry* hist, size_t nb) {
    hist_scalar(b,g,h,r,n,hist,nb);
}
#endif

// ---- Multi-threaded histogram: partition features across threads ----
void mt_histogram_features(
    const std::vector<uint8_t>& col_bins,  // col-major: [col * n_rows + row]
    const double* grads, const double* hess,
    const uint32_t* rows, size_t n_rows, size_t n_cols,
    size_t n_bins, int n_threads, bool use_simd,
    std::vector<std::vector<HistEntry>>& all_hists)
{
    all_hists.resize(n_cols, std::vector<HistEntry>(n_bins));
    
    auto worker = [&](int tid) {
        size_t f_start = (size_t)tid * n_cols / n_threads;
        size_t f_end   = (size_t)(tid+1) * n_cols / n_threads;
        for (size_t f = f_start; f < f_end; ++f) {
            const uint8_t* col = &col_bins[f * n_rows];
            if (use_simd)
                hist_neon(col, grads, hess, rows, n_rows, all_hists[f].data(), n_bins);
            else
                hist_scalar(col, grads, hess, rows, n_rows, all_hists[f].data(), n_bins);
        }
    };
    
    if (n_threads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> threads;
        for (int t = 0; t < n_threads; ++t)
            threads.emplace_back(worker, t);
        for (auto& t : threads) t.join();
    }
}

int main() {
    // Generate Friedman #1 data
    const size_t N = 100000, D = 50, NBINS = 256;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> unif(0, 1);
    std::normal_distribution<double> noise(0, 1);
    
    std::vector<float> X(N * D);
    std::vector<double> y(N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < D; ++j) X[i*D+j] = unif(rng);
        y[i] = 10*sin(M_PI*X[i*D]*X[i*D+1]) + 20*pow(X[i*D+2]-0.5,2)
             + 10*X[i*D+3] + 5*X[i*D+4] + noise(rng);
    }
    
    // Bin mapping
    BinMapper mapper;
    mapper.fit(X.data(), N, D, NBINS);
    
    // Create column-major binned data
    std::vector<uint8_t> col_bins(N * D);
    for (size_t c = 0; c < D; ++c)
        for (size_t r = 0; r < N; ++r)
            col_bins[c * N + r] = mapper.map(c, X[r * D + c]);
    
    std::vector<double> grads(N, 0.1), hess(N, 1.0);
    std::vector<uint32_t> rows(N);
    std::iota(rows.begin(), rows.end(), 0u);
    
    std::cout << "[";
    bool first = true;
    
    for (int n_threads : {1, 2, 4, 8}) {
        for (bool use_simd : {false, true}) {
            std::vector<std::vector<HistEntry>> all_hists;
            
            // Warmup
            mt_histogram_features(col_bins, grads.data(), hess.data(),
                                  rows.data(), N, D, NBINS, n_threads, use_simd, all_hists);
            
            const int ITERS = 50;
            auto t0 = Clock::now();
            for (int it = 0; it < ITERS; ++it)
                mt_histogram_features(col_bins, grads.data(), hess.data(),
                                      rows.data(), N, D, NBINS, n_threads, use_simd, all_hists);
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / ITERS;
            
            if (!first) std::cout << ", ";
            first = false;
            std::cout << std::fixed << std::setprecision(2)
                      << "{\"threads\": " << n_threads
                      << ", \"simd\": " << (use_simd ? "true" : "false")
                      << ", \"time_ms\": " << ms << "}";
        }
    }
    std::cout << "]" << std::endl;
    
    return 0;
}
'''


def compile_and_run(name, src, args="", timeout=120):
    """Compile C++ source and run with given args."""
    src_path = f"/tmp/{name}.cpp"
    bin_path = f"/tmp/{name}"
    
    with open(src_path, "w") as f:
        f.write(src)
    
    ret = subprocess.run(
        ["clang++", "-std=c++17", "-O3", "-mcpu=apple-m3", "-pthread",
         "-o", bin_path, src_path],
        capture_output=True, text=True
    )
    if ret.returncode != 0:
        print(f"  Compile error for {name}: {ret.stderr[:300]}")
        return None
    
    ret = subprocess.run(
        [bin_path] + (args.split() if args else []),
        capture_output=True, text=True, timeout=timeout
    )
    if ret.returncode != 0:
        print(f"  Runtime error for {name}: {ret.stderr[:200]}")
        return None
    
    return ret.stdout.strip()


# ============================================================================
#  PART 3: XGBoost Kernel Breakdown
# ============================================================================
def xgboost_kernel_breakdown():
    """Profile XGBoost training to breakdown where time goes."""
    import xgboost as xgb
    
    print("\n[XGBoost Kernel Breakdown]")
    
    def gen_friedman(n, d=20, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.uniform(0, 1, (n, d)).astype(np.float32)
        y = (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2
             + 10*X[:,3] + 5*X[:,4] + rng.normal(0, 1, n))
        return X, y
    
    results = {}
    
    for n_samples in [10000, 50000, 100000]:
        X, y = gen_friedman(n_samples, 20)
        X_test, y_test = gen_friedman(5000, 20, seed=99)
        
        dtrain = xgb.DMatrix(X, label=y)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        params = {
            "max_depth": 6, "learning_rate": 0.1, "reg_lambda": 1.0,
            "tree_method": "hist", "max_bin": 256,
            "nthread": 1, "verbosity": 0, "objective": "reg:squarederror",
        }
        
        # Total training time
        t0 = time.perf_counter()
        model = xgb.train(params, dtrain, num_boost_round=50)
        total_ms = (time.perf_counter() - t0) * 1000
        
        # Inference time
        t0 = time.perf_counter()
        preds = model.predict(dtest)
        infer_ms = (time.perf_counter() - t0) * 1000
        
        # Per-tree timing (approximate by training 1 tree at a time)
        per_tree_times = []
        for n_trees in [1, 10, 20, 30, 40, 50]:
            t0 = time.perf_counter()
            xgb.train(params, dtrain, num_boost_round=n_trees)
            tree_ms = (time.perf_counter() - t0) * 1000
            per_tree_times.append({"n_trees": n_trees, "time_ms": round(tree_ms, 2)})
        
        # Data preparation time
        t0 = time.perf_counter()
        _ = xgb.DMatrix(X, label=y)
        data_prep_ms = (time.perf_counter() - t0) * 1000
        
        # Compute per-tree marginal cost
        if len(per_tree_times) >= 2:
            t50 = per_tree_times[-1]["time_ms"]
            t1 = per_tree_times[0]["time_ms"]
            marginal_per_tree = (t50 - t1) / 49.0
            overhead = t1  # first tree includes setup
        else:
            marginal_per_tree = total_ms / 50
            overhead = 0
        
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        
        result = {
            "n_samples": n_samples,
            "total_train_ms": round(total_ms, 2),
            "data_prep_ms": round(data_prep_ms, 2),
            "infer_ms": round(infer_ms, 2),
            "marginal_per_tree_ms": round(marginal_per_tree, 2),
            "setup_overhead_ms": round(overhead, 2),
            "rmse": round(float(rmse), 6),
            "per_tree_curve": per_tree_times,
        }
        results[str(n_samples)] = result
        
        print(f"  N={n_samples}: total={total_ms:.1f}ms, "
              f"per_tree={marginal_per_tree:.2f}ms, "
              f"data_prep={data_prep_ms:.1f}ms, rmse={rmse:.4f}")
    
    # Thread scaling
    print("\n  [XGBoost Thread Scaling]")
    X, y = gen_friedman(100000, 20)
    thread_results = []
    for nt in [1, 2, 4, 8]:
        params["nthread"] = nt
        dtrain = xgb.DMatrix(X, label=y)
        t0 = time.perf_counter()
        xgb.train(params, dtrain, num_boost_round=50)
        ms = (time.perf_counter() - t0) * 1000
        thread_results.append({"threads": nt, "time_ms": round(ms, 2)})
        print(f"    {nt} threads: {ms:.1f} ms")
    
    results["thread_scaling"] = thread_results
    return results


# ============================================================================
#  Figure Generation
# ============================================================================
def generate_figures(hw_results, mt_results, xgb_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
    })
    
    # ---- Fig 11: Cache working set analysis ----
    if "cache_working_set" in hw_results:
        d = hw_results["cache_working_set"]["data"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ws = [x["ws_kb"] for x in d]
        scalar = [x["scalar_ns_elem"] for x in d]
        neon = [x["neon_ns_elem"] for x in d]
        
        ax.plot(ws, scalar, 'o-', color='#3498db', linewidth=2, markersize=6, label='Scalar')
        ax.plot(ws, neon, 's-', color='#2ecc71', linewidth=2, markersize=6, label='NEON')
        
        # Cache boundaries
        ax.axvline(x=64, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axvline(x=4096, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(64, max(scalar)*0.9, ' L1d\n 64KB', color='red', fontsize=9, fontweight='bold')
        ax.text(4096, max(scalar)*0.9, ' L2\n 4MB', color='orange', fontsize=9, fontweight='bold')
        
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Working Set Size (KB)')
        ax.set_ylabel('Time per Element (ns)')
        ax.set_title('Cache Effects on Histogram Construction\n(Apple M3: L1d=64KB, L2=4MB)')
        ax.legend()
        plt.savefig(f'{FIGURES_DIR}/fig11_cache_analysis.png')
        plt.savefig(f'{FIGURES_DIR}/fig11_cache_analysis.pdf')
        print("  ✓ fig11_cache_analysis")
        plt.close()
    
    # ---- Fig 12: Branch prediction / bin collision ----
    if "branch_prediction" in hw_results:
        d = hw_results["branch_prediction"]["data"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        nbins = [x["bins"] for x in d]
        collision = [x["collision_rate"] * 100 for x in d]
        speedups = [x["speedup"] for x in d]
        scalar_t = [x["scalar_us"] for x in d]
        neon_t = [x["neon_us"] for x in d]
        
        ax1.bar(range(len(nbins)), collision, color='#9b59b6', edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Number of Unique Bins')
        ax1.set_ylabel('4-Way Collision Rate (%)')
        ax1.set_title('Bin Collision Rate\n(4 consecutive rows same bin)')
        ax1.set_xticks(range(len(nbins)))
        ax1.set_xticklabels(nbins)
        
        for i, (v, s) in enumerate(zip(collision, speedups)):
            ax1.text(i, v + 1, f'{s:.2f}×', ha='center', fontsize=8, fontweight='bold',
                     color='#2ecc71' if s > 1.0 else '#e74c3c')
        
        x = np.arange(len(nbins))
        ax2.bar(x - 0.15, scalar_t, 0.3, label='Scalar', color='#3498db', edgecolor='black', linewidth=0.5)
        ax2.bar(x + 0.15, neon_t, 0.3, label='NEON', color='#2ecc71', edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Number of Unique Bins')
        ax2.set_ylabel('Histogram Time (µs)')
        ax2.set_title('NEON Speedup vs Bin Entropy')
        ax2.set_xticks(x)
        ax2.set_xticklabels(nbins)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/fig12_branch_prediction.png')
        plt.savefig(f'{FIGURES_DIR}/fig12_branch_prediction.pdf')
        print("  ✓ fig12_branch_prediction")
        plt.close()
    
    # ---- Fig 13: IPC / Cycles-per-element ----
    if "ipc_estimation" in hw_results:
        d = hw_results["ipc_estimation"]["data"]
        fig, ax = plt.subplots(figsize=(7, 5))
        
        kernels = [x["kernel"] for x in d]
        scalar_cpe = [x["scalar_cpe"] for x in d]
        neon_cpe = [x["neon_cpe"] for x in d]
        
        x = np.arange(len(kernels))
        ax.bar(x - 0.15, scalar_cpe, 0.3, label='Scalar', color='#3498db', edgecolor='black', linewidth=0.5)
        ax.bar(x + 0.15, neon_cpe, 0.3, label='NEON', color='#2ecc71', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Kernel')
        ax.set_ylabel('Cycles per Element')
        ax.set_title('Cycles per Element by Kernel\n(Apple M3 timer counter)')
        ax.set_xticks(x)
        labels = ["Histogram\n(scatter-bound)" if k == "histogram" else "Gradient\n(map-parallel)" for k in kernels]
        ax.set_xticklabels(labels)
        ax.legend()
        
        for i in range(len(kernels)):
            speedup = scalar_cpe[i] / neon_cpe[i] if neon_cpe[i] > 0 else 1
            ax.text(i, max(scalar_cpe[i], neon_cpe[i]) * 1.05,
                    f'{speedup:.2f}×', ha='center', fontsize=10, fontweight='bold',
                    color='#2ecc71' if speedup > 1 else '#e74c3c')
        
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/fig13_cycles_per_element.png')
        plt.savefig(f'{FIGURES_DIR}/fig13_cycles_per_element.pdf')
        print("  ✓ fig13_cycles_per_element")
        plt.close()
    
    # ---- Fig 14: Multi-threading scaling ----
    if mt_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        scalar_data = [x for x in mt_results if not x.get("simd", False)]
        neon_data = [x for x in mt_results if x.get("simd", False)]
        
        threads_s = [x["threads"] for x in scalar_data]
        times_s = [x["time_ms"] for x in scalar_data]
        threads_n = [x["threads"] for x in neon_data]
        times_n = [x["time_ms"] for x in neon_data]
        
        # Absolute times
        ax1.plot(threads_s, times_s, 'o-', color='#3498db', linewidth=2, markersize=8, label='Scalar')
        ax1.plot(threads_n, times_n, 's-', color='#2ecc71', linewidth=2, markersize=8, label='NEON')
        ax1.set_xlabel('Number of Threads')
        ax1.set_ylabel('Histogram Construction Time (ms)')
        ax1.set_title('Thread Scaling: Histogram Construction\n(100K samples, 50 features)')
        ax1.set_xticks([1, 2, 4, 8])
        ax1.legend()
        
        for t, ms in zip(threads_s, times_s):
            ax1.annotate(f'{ms:.1f}', (t, ms), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8)
        for t, ms in zip(threads_n, times_n):
            ax1.annotate(f'{ms:.1f}', (t, ms), textcoords="offset points",
                         xytext=(0, -15), ha='center', fontsize=8, color='#2ecc71')
        
        # Speedup
        if times_s[0] > 0 and times_n[0] > 0:
            speedup_s = [times_s[0] / t for t in times_s]
            speedup_n = [times_n[0] / t for t in times_n]
            ideal = [1, 2, 4, 8]
            
            ax2.plot(threads_s, speedup_s, 'o-', color='#3498db', linewidth=2, markersize=8, label='Scalar')
            ax2.plot(threads_n, speedup_n, 's-', color='#2ecc71', linewidth=2, markersize=8, label='NEON')
            ax2.plot([1, 2, 4, 8], ideal, '--', color='gray', alpha=0.5, label='Ideal')
            ax2.set_xlabel('Number of Threads')
            ax2.set_ylabel('Speedup (vs 1 thread)')
            ax2.set_title('Thread Scaling Efficiency')
            ax2.set_xticks([1, 2, 4, 8])
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/fig14_thread_scaling.png')
        plt.savefig(f'{FIGURES_DIR}/fig14_thread_scaling.pdf')
        print("  ✓ fig14_thread_scaling")
        plt.close()
    
    # ---- Fig 15: XGBoost thread scaling comparison ----
    if xgb_results and "thread_scaling" in xgb_results:
        xgb_ts = xgb_results["thread_scaling"]
        
        fig, ax = plt.subplots(figsize=(7, 5))
        xgb_threads = [x["threads"] for x in xgb_ts]
        xgb_times = [x["time_ms"] for x in xgb_ts]
        xgb_speedup = [xgb_times[0] / t for t in xgb_times]
        
        # Add C++ data if available (using scalar 1-thread as base)
        if mt_results:
            scalar_1t = next((x["time_ms"] for x in mt_results if x["threads"]==1 and not x["simd"]), None)
            neon_1t = next((x["time_ms"] for x in mt_results if x["threads"]==1 and x["simd"]), None)
            # Note: mt_results is histogram-only, xgb is full training. Just show XGB scaling.
        
        ax.plot(xgb_threads, xgb_speedup, 'o-', color='#e74c3c', linewidth=2, label='XGBoost')
        ax.plot([1,2,4,8], [1,2,4,8], '--', color='gray', alpha=0.5, label='Ideal')
        ax.set_xlabel('Threads')
        ax.set_ylabel('Speedup')
        ax.set_title('XGBoost Thread Scaling\n(100K samples, 20 features, 50 trees)')
        ax.set_xticks([1, 2, 4, 8])
        ax.legend()
        
        for t, s, ms in zip(xgb_threads, xgb_speedup, xgb_times):
            ax.annotate(f'{s:.1f}× ({ms:.0f}ms)', (t, s),
                        textcoords="offset points", xytext=(5, 10), fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/fig15_xgboost_scaling.png')
        plt.savefig(f'{FIGURES_DIR}/fig15_xgboost_scaling.pdf')
        print("  ✓ fig15_xgboost_scaling")
        plt.close()
    
    # ---- Fig 16: XGBoost per-tree cost breakdown ----
    if xgb_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ["10000", "50000", "100000"]:
            if key in xgb_results:
                curve = xgb_results[key]["per_tree_curve"]
                nts = [x["n_trees"] for x in curve]
                tms = [x["time_ms"] for x in curve]
                ax.plot(nts, tms, 'o-', linewidth=2, markersize=6, label=f'N={key}')
        
        ax.set_xlabel('Number of Trees')
        ax.set_ylabel('Total Training Time (ms)')
        ax.set_title('XGBoost Training Time vs Trees\n(breakdown: setup + marginal per tree)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/fig16_xgboost_breakdown.png')
        plt.savefig(f'{FIGURES_DIR}/fig16_xgboost_breakdown.pdf')
        print("  ✓ fig16_xgboost_breakdown")
        plt.close()


# ============================================================================
#  Main
# ============================================================================
def main():
    print("=" * 70)
    print("  Advanced Benchmarks: HW Counters, Multi-Threading, XGBoost Breakdown")
    print(f"  Platform: Apple M3 (4P+4E), 16GB, L1d=64KB, L2=4MB")
    print("=" * 70)
    
    all_hw_results = {}
    
    # ---- Part 1: Hardware Counter Micro-Benchmarks ----
    print("\n[1] Hardware Counter Micro-Benchmarks")
    
    for bench in ["cache", "branch", "ipc", "access"]:
        print(f"\n  Running {bench} benchmark...")
        output = compile_and_run("hw_counters_bench", HW_COUNTER_SRC, bench)
        if output:
            try:
                parsed = json.loads(output)
                all_hw_results[parsed["bench"]] = parsed
                print(f"    ✓ {bench} complete")
            except json.JSONDecodeError:
                print(f"    ✗ Failed to parse output: {output[:100]}")
    
    # ---- Part 2: Multi-Threading Scaling ----
    print("\n\n[2] Multi-Threading Scaling")
    mt_results = None
    output = compile_and_run("mt_bench", MT_BENCH_SRC, "", timeout=180)
    if output:
        try:
            mt_results = json.loads(output)
            print("  Thread scaling results:")
            for r in mt_results:
                mode = "NEON" if r.get("simd") else "Scalar"
                print(f"    {r['threads']}T {mode}: {r['time_ms']:.1f} ms")
        except json.JSONDecodeError:
            print(f"  Failed to parse: {output[:200]}")
    
    # ---- Part 3: XGBoost Kernel Breakdown ----
    print("\n\n[3] XGBoost Kernel Breakdown")
    xgb_results = xgboost_kernel_breakdown()
    
    # ---- Part 4: Generate Figures ----
    print("\n\n[4] Generating figures...")
    generate_figures(all_hw_results, mt_results, xgb_results)
    
    # ---- Save all results ----
    all_results = {
        "hw_counters": all_hw_results,
        "mt_scaling": mt_results,
        "xgboost_breakdown": xgb_results,
        "platform": {
            "cpu": "Apple M3",
            "cores": "4P + 4E = 8",
            "l1d": "64 KB",
            "l2": "4 MB",
            "cache_line": "128 bytes",
            "memory": "16 GB unified",
        }
    }
    
    results_file = f"{RESULTS_DIR}/advanced_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_file}")
    
    print(f"\n{'=' * 70}")
    print(f"  All advanced benchmarks complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
