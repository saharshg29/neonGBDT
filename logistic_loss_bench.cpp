/*
 * logistic_loss_bench.cpp — Squared vs Logistic loss gradient SIMD benchmark
 * Uses higher precision exp approximation and nanosecond timing.
 */
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#define HAS_NEON 1
#endif

using Clock = std::chrono::high_resolution_clock;

// ============================================================================
//  Squared Loss
// ============================================================================
void squared_loss_scalar(const double* preds, const double* labels,
                         double* grads, double* hess, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        grads[i] = preds[i] - labels[i];
        hess[i] = 1.0;
    }
}

#if HAS_NEON
void squared_loss_neon(const double* preds, const double* labels,
                       double* grads, double* hess, size_t n) {
    const float64x2_t ones = vdupq_n_f64(1.0);
    size_t i = 0;
    const size_t n4 = n & ~size_t(3);
    for (; i < n4; i += 4) {
        float64x2_t p0 = vld1q_f64(&preds[i]);
        float64x2_t l0 = vld1q_f64(&labels[i]);
        float64x2_t p1 = vld1q_f64(&preds[i+2]);
        float64x2_t l1 = vld1q_f64(&labels[i+2]);
        vst1q_f64(&grads[i],   vsubq_f64(p0, l0));
        vst1q_f64(&grads[i+2], vsubq_f64(p1, l1));
        vst1q_f64(&hess[i],   ones);
        vst1q_f64(&hess[i+2], ones);
    }
    for (; i < n; ++i) {
        grads[i] = preds[i] - labels[i];
        hess[i] = 1.0;
    }
}
#endif

// ============================================================================
//  Logistic Loss
// ============================================================================
static inline double sigmoid_scalar(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

void logistic_loss_scalar(const double* preds, const double* labels,
                          double* grads, double* hess, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        double p = sigmoid_scalar(preds[i]);
        grads[i] = p - labels[i];
        hess[i] = std::max(p * (1.0 - p), 1e-16);
    }
}

#if HAS_NEON
// High-accuracy NEON sigmoid: just use scalar exp per lane
// This tests if SIMD helps even with scalar exp fallback
static inline float64x2_t sigmoid_neon_accurate(float64x2_t x) {
    double x0 = vgetq_lane_f64(x, 0);
    double x1 = vgetq_lane_f64(x, 1);
    double s0 = 1.0 / (1.0 + std::exp(-x0));
    double s1 = 1.0 / (1.0 + std::exp(-x1));
    return float64x2_t{s0, s1};
}

void logistic_loss_neon(const double* preds, const double* labels,
                        double* grads, double* hess, size_t n) {
    const float64x2_t ones = vdupq_n_f64(1.0);
    const float64x2_t eps = vdupq_n_f64(1e-16);
    size_t i = 0;
    const size_t n4 = n & ~size_t(3);
    for (; i < n4; i += 4) {
        float64x2_t p0 = vld1q_f64(&preds[i]);
        float64x2_t l0 = vld1q_f64(&labels[i]);
        float64x2_t p1 = vld1q_f64(&preds[i+2]);
        float64x2_t l1 = vld1q_f64(&labels[i+2]);

        float64x2_t sig0 = sigmoid_neon_accurate(p0);
        float64x2_t sig1 = sigmoid_neon_accurate(p1);

        vst1q_f64(&grads[i],   vsubq_f64(sig0, l0));
        vst1q_f64(&grads[i+2], vsubq_f64(sig1, l1));

        float64x2_t h0 = vmulq_f64(sig0, vsubq_f64(ones, sig0));
        float64x2_t h1 = vmulq_f64(sig1, vsubq_f64(ones, sig1));
        vst1q_f64(&hess[i],   vmaxq_f64(h0, eps));
        vst1q_f64(&hess[i+2], vmaxq_f64(h1, eps));
    }
    for (; i < n; ++i) {
        double s = sigmoid_scalar(preds[i]);
        grads[i] = s - labels[i];
        hess[i] = std::max(s * (1.0 - s), 1e-16);
    }
}
#endif

// ============================================================================
//  Combined gradient+histogram benchmark (simulates real training iteration)
// ============================================================================
struct HistEntry {
    double grad_sum=0, hess_sum=0; uint32_t count=0, pad=0;
    void clear() { grad_sum=0; hess_sum=0; count=0; }
};

void build_hist(const uint8_t* bins, const double* g, const double* h,
                const uint32_t* rows, size_t n, HistEntry* hist, size_t nb) {
    for (size_t i = 0; i < nb; ++i) hist[i].clear();
    for (size_t i = 0; i < n; ++i) {
        uint32_t r = rows[i]; uint8_t b = bins[r];
        hist[b].grad_sum += g[r]; hist[b].hess_sum += h[r]; hist[b].count++;
    }
}

int main() {
    const size_t N = 200000;
    const size_t NBINS = 256;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> unif(-3.0, 3.0);
    std::uniform_int_distribution<int> label_dist(0, 1);

    std::vector<double> preds(N), labels_reg(N), labels_cls(N);
    std::vector<double> grads(N), hess_vec(N);
    std::vector<uint8_t> bins(N);
    std::vector<uint32_t> rows(N);
    std::vector<HistEntry> hist(NBINS);

    for (size_t i = 0; i < N; ++i) {
        preds[i] = unif(rng);
        labels_reg[i] = unif(rng);
        labels_cls[i] = (double)label_dist(rng);
        bins[i] = rng() % NBINS;
        rows[i] = (uint32_t)i;
    }

    const int ITERS = 500;
    std::cout << "[";

    // ---- Squared Loss: Scalar ----
    {
        // warmup
        squared_loss_scalar(preds.data(), labels_reg.data(), grads.data(), hess_vec.data(), N);
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it)
            squared_loss_scalar(preds.data(), labels_reg.data(), grads.data(), hess_vec.data(), N);
        double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / ITERS / N;
        std::cout << std::fixed << std::setprecision(3)
                  << "{\"loss\": \"squared\", \"impl\": \"scalar\", \"ns_per_elem\": " << ns << "}";
    }

#if HAS_NEON
    // ---- Squared Loss: NEON ----
    {
        squared_loss_neon(preds.data(), labels_reg.data(), grads.data(), hess_vec.data(), N);
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it)
            squared_loss_neon(preds.data(), labels_reg.data(), grads.data(), hess_vec.data(), N);
        double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / ITERS / N;
        std::cout << ", " << std::fixed << std::setprecision(3)
                  << "{\"loss\": \"squared\", \"impl\": \"neon\", \"ns_per_elem\": " << ns << "}";
    }
#endif

    // ---- Logistic Loss: Scalar ----
    {
        logistic_loss_scalar(preds.data(), labels_cls.data(), grads.data(), hess_vec.data(), N);
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it)
            logistic_loss_scalar(preds.data(), labels_cls.data(), grads.data(), hess_vec.data(), N);
        double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / ITERS / N;
        std::cout << ", " << std::fixed << std::setprecision(3)
                  << "{\"loss\": \"logistic\", \"impl\": \"scalar\", \"ns_per_elem\": " << ns << "}";
    }

#if HAS_NEON
    // ---- Logistic Loss: NEON ----
    {
        logistic_loss_neon(preds.data(), labels_cls.data(), grads.data(), hess_vec.data(), N);
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it)
            logistic_loss_neon(preds.data(), labels_cls.data(), grads.data(), hess_vec.data(), N);
        double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / ITERS / N;
        std::cout << ", " << std::fixed << std::setprecision(3)
                  << "{\"loss\": \"logistic\", \"impl\": \"neon\", \"ns_per_elem\": " << ns << "}";
    }
#endif

    // ---- Combined: grad+hist per iteration (simulates one tree level) ----
    {
        // Squared scalar
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            squared_loss_scalar(preds.data(), labels_reg.data(), grads.data(), hess_vec.data(), N);
            build_hist(bins.data(), grads.data(), hess_vec.data(), rows.data(), N, hist.data(), NBINS);
        }
        double sq_sc_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / ITERS;

        // Logistic scalar
        t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            logistic_loss_scalar(preds.data(), labels_cls.data(), grads.data(), hess_vec.data(), N);
            build_hist(bins.data(), grads.data(), hess_vec.data(), rows.data(), N, hist.data(), NBINS);
        }
        double log_sc_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / ITERS;

#if HAS_NEON
        // Logistic NEON + scalar hist
        t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            logistic_loss_neon(preds.data(), labels_cls.data(), grads.data(), hess_vec.data(), N);
            build_hist(bins.data(), grads.data(), hess_vec.data(), rows.data(), N, hist.data(), NBINS);
        }
        double log_neon_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / ITERS;
#else
        double log_neon_us = log_sc_us;
#endif

        std::cout << ", " << std::fixed << std::setprecision(1)
                  << "{\"combined\": true"
                  << ", \"sq_scalar_us\": " << sq_sc_us
                  << ", \"log_scalar_us\": " << log_sc_us
                  << ", \"log_neon_us\": " << log_neon_us
                  << ", \"logistic_overhead_pct\": " << std::setprecision(1) << ((log_sc_us / sq_sc_us - 1.0) * 100)
                  << ", \"neon_benefit_pct\": " << std::setprecision(1) << ((1.0 - log_neon_us / log_sc_us) * 100)
                  << "}";
    }

    // Verify correctness
    {
        std::vector<double> g_sc(N), h_sc(N), g_ne(N), h_ne(N);
        logistic_loss_scalar(preds.data(), labels_cls.data(), g_sc.data(), h_sc.data(), N);
#if HAS_NEON
        logistic_loss_neon(preds.data(), labels_cls.data(), g_ne.data(), h_ne.data(), N);
#else
        g_ne = g_sc; h_ne = h_sc;
#endif
        double max_ge = 0, max_he = 0;
        for (size_t i = 0; i < N; ++i) {
            max_ge = std::max(max_ge, std::abs(g_sc[i] - g_ne[i]));
            max_he = std::max(max_he, std::abs(h_sc[i] - h_ne[i]));
        }
        std::cout << ", " << std::scientific << std::setprecision(2)
                  << "{\"verification\": true, \"max_grad_err\": " << max_ge
                  << ", \"max_hess_err\": " << max_he << "}";
    }

    std::cout << "]" << std::endl;
    return 0;
}
