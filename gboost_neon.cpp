/*
 * ============================================================================
 *  Compiler-Aware Gradient Boosting with SIMD Histogram Binning,
 *  Sparsity Handling, and Code Generation
 *
 *  *** ARM NEON version (Apple Silicon / AArch64) ***
 *
 *  Compile (Apple Silicon):
 *    g++ -std=c++17 -O3 -mcpu=apple-m1 -o gboost gboost_neon.cpp
 *    clang++ -std=c++17 -O3 -mcpu=apple-m1 -o gboost gboost_neon.cpp
 *
 *  With LTO:
 *    clang++ -std=c++17 -O3 -mcpu=apple-m1 -flto -o gboost gboost_neon.cpp
 *
 *  Profile (macOS):
 *    xcrun xctrace record --template "Time Profiler" --launch ./gboost
 *    instruments -t "Time Profiler" ./gboost
 * ============================================================================
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

// ---- Platform-specific SIMD headers ----
#if defined(__aarch64__) || defined(_M_ARM64)
#   include <arm_neon.h>
#   define HAS_NEON 1
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#   ifdef _MSC_VER
#       include <intrin.h>
#   else
#       include <x86intrin.h>
#   endif
#   if defined(__AVX2__)
#       define HAS_AVX2 1
#   endif
#   if defined(__SSE2__)
#       define HAS_SSE2 1
#   endif
#else
#   define SCALAR_ONLY 1
#endif

// ============================================================================
//  SECTION 1: CONFIGURATION & TIMER
// ============================================================================

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
    std::string label_;
public:
    Timer(const std::string& label = "") : t0_(Clock::now()), label_(label) {}
    void reset() { t0_ = Clock::now(); }
    double elapsed_us()  const {
        return std::chrono::duration<double, std::micro>(
            Clock::now() - t0_).count();
    }
    double elapsed_ms()  const { return elapsed_us() / 1000.0; }
    ~Timer() {
        if (!label_.empty())
            std::cout << "  [TIMER] " << label_ << ": "
                      << std::fixed << std::setprecision(2)
                      << elapsed_ms() << " ms\n";
    }
};

// ============================================================================
//  SECTION 2: SIMD HELPERS — ARM NEON
// ============================================================================

#if HAS_NEON

// Horizontal sum of float64x2_t → double
static inline double hsum_f64x2(float64x2_t v) {
    return vgetq_lane_f64(v, 0) + vgetq_lane_f64(v, 1);
}

// Horizontal sum of float32x4_t → float
static inline float hsum_f32x4(float32x4_t v) {
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t sum = vadd_f32(lo, hi);             // [a+c, b+d]
    return vget_lane_f32(vpadd_f32(sum, sum), 0);    // a+b+c+d
}

#endif // HAS_NEON

// ============================================================================
//  SECTION 3: SPARSE DATA (CSR FORMAT)
// ============================================================================

struct CSRMatrix {
    std::vector<float>    values;
    std::vector<uint32_t> col_indices;
    std::vector<size_t>   row_ptr;
    size_t n_rows = 0;
    size_t n_cols = 0;
    double sparsity = 0.0;

    float get(size_t row, size_t col) const {
        for (size_t j = row_ptr[row]; j < row_ptr[row + 1]; ++j)
            if (col_indices[j] == col) return values[j];
        return 0.0f;
    }
    size_t nnz() const { return values.size(); }
};

struct DenseMatrix {
    std::vector<float> data;
    size_t n_rows = 0;
    size_t n_cols = 0;

    float  get(size_t r, size_t c) const { return data[r * n_cols + c]; }
    float& get(size_t r, size_t c)       { return data[r * n_cols + c]; }
    const float* row_ptr(size_t r) const { return &data[r * n_cols]; }
};

CSRMatrix to_csr(const DenseMatrix& dm) {
    CSRMatrix csr;
    csr.n_rows = dm.n_rows;
    csr.n_cols = dm.n_cols;
    csr.row_ptr.reserve(dm.n_rows + 1);
    csr.row_ptr.push_back(0);

    size_t total = dm.n_rows * dm.n_cols;
    size_t nnz_count = 0;

    for (size_t r = 0; r < dm.n_rows; ++r) {
        for (size_t c = 0; c < dm.n_cols; ++c) {
            float v = dm.get(r, c);
            if (v != 0.0f) {
                csr.values.push_back(v);
                csr.col_indices.push_back(static_cast<uint32_t>(c));
                ++nnz_count;
            }
        }
        csr.row_ptr.push_back(csr.values.size());
    }

    csr.sparsity = 1.0 -
        static_cast<double>(nnz_count) / static_cast<double>(total);
    return csr;
}

// ============================================================================
//  SECTION 4: HISTOGRAM BINNING
// ============================================================================

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

// Column-major binned data (cache-friendly for histogram building)
struct ColumnBinnedData {
    std::vector<uint8_t> bins;   // [col * n_rows + row]
    size_t n_rows, n_cols;
    size_t max_bins;

    const uint8_t* col(size_t c) const { return &bins[c * n_rows]; }
};

ColumnBinnedData create_binned_data(const DenseMatrix& X,
                                     const BinMapper& mapper) {
    ColumnBinnedData bd;
    bd.n_rows   = X.n_rows;
    bd.n_cols   = X.n_cols;
    bd.max_bins = mapper.max_bins;
    bd.bins.resize(X.n_rows * X.n_cols);

    for (size_t c = 0; c < X.n_cols; ++c)
        for (size_t r = 0; r < X.n_rows; ++r)
            bd.bins[c * X.n_rows + r] = mapper.map_value(c, X.get(r, c));

    return bd;
}

// ============================================================================
//  SECTION 5: HISTOGRAM STRUCTURES & SIMD CONSTRUCTION
// ============================================================================

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

// ---------------------------------------------------------------------------
//  5a. SCALAR histogram construction
// ---------------------------------------------------------------------------
void build_histogram_scalar(
    const uint8_t*    bin_col,
    const double*     gradients,
    const double*     hessians,
    const uint32_t*   row_indices,
    size_t            n_rows,
    Histogram&        hist)
{
    hist.clear();
    for (size_t i = 0; i < n_rows; ++i) {
        uint32_t row = row_indices[i];
        uint8_t  bin = bin_col[row];
        hist.entries[bin].grad_sum += gradients[row];
        hist.entries[bin].hess_sum += hessians[row];
        hist.entries[bin].count    += 1;
    }
}

// ---------------------------------------------------------------------------
//  5b. NEON histogram construction
//      Strategy: prefetch rows in batches of 4, load grad/hess with NEON,
//      accumulate into per-bin accumulators. The scatter to bins is inherently
//      serial, but we SIMD-ize the gather of gradient/hessian pairs.
// ---------------------------------------------------------------------------
#if HAS_NEON

void build_histogram_simd(
    const uint8_t*    bin_col,
    const double*     gradients,
    const double*     hessians,
    const uint32_t*   row_indices,
    size_t            n_rows,
    Histogram&        hist)
{
    hist.clear();

    if (n_rows < 16) {
        build_histogram_scalar(bin_col, gradients, hessians,
                               row_indices, n_rows, hist);
        return;
    }

    // Process 4 rows at a time: gather grad/hess, scatter to bins
    size_t i = 0;
    const size_t n4 = n_rows & ~size_t(3);

    for (; i < n4; i += 4) {
        // Prefetch next cache lines
        __builtin_prefetch(&row_indices[i + 16], 0, 1);

        uint32_t r0 = row_indices[i];
        uint32_t r1 = row_indices[i + 1];
        uint32_t r2 = row_indices[i + 2];
        uint32_t r3 = row_indices[i + 3];

        uint8_t b0 = bin_col[r0];
        uint8_t b1 = bin_col[r1];
        uint8_t b2 = bin_col[r2];
        uint8_t b3 = bin_col[r3];

        // Load gradient pairs with NEON
        float64x2_t g01 = {gradients[r0], gradients[r1]};
        float64x2_t g23 = {gradients[r2], gradients[r3]};
        float64x2_t h01 = {hessians[r0],  hessians[r1]};
        float64x2_t h23 = {hessians[r2],  hessians[r3]};

        // If consecutive rows map to same bin → NEON add
        if (b0 == b1 && b2 == b3 && b0 == b2) {
            // All 4 go to same bin — sum with NEON
            float64x2_t gsum = vaddq_f64(g01, g23);
            float64x2_t hsum = vaddq_f64(h01, h23);
            hist.entries[b0].grad_sum += hsum_f64x2(gsum);
            hist.entries[b0].hess_sum += hsum_f64x2(hsum);
            hist.entries[b0].count    += 4;
        } else if (b0 == b1) {
            // First pair same bin
            float64x2_t gsum = {vgetq_lane_f64(g01, 0) + vgetq_lane_f64(g01, 1), 0};
            hist.entries[b0].grad_sum += vgetq_lane_f64(gsum, 0);
            hist.entries[b0].hess_sum += hsum_f64x2(h01);
            hist.entries[b0].count    += 2;
            // Remaining two scalar
            hist.entries[b2].grad_sum += gradients[r2];
            hist.entries[b2].hess_sum += hessians[r2];
            hist.entries[b2].count    += 1;
            hist.entries[b3].grad_sum += gradients[r3];
            hist.entries[b3].hess_sum += hessians[r3];
            hist.entries[b3].count    += 1;
        } else {
            // All different bins — scalar scatter
            hist.entries[b0].grad_sum += gradients[r0];
            hist.entries[b0].hess_sum += hessians[r0];
            hist.entries[b0].count    += 1;

            hist.entries[b1].grad_sum += gradients[r1];
            hist.entries[b1].hess_sum += hessians[r1];
            hist.entries[b1].count    += 1;

            hist.entries[b2].grad_sum += gradients[r2];
            hist.entries[b2].hess_sum += hessians[r2];
            hist.entries[b2].count    += 1;

            hist.entries[b3].grad_sum += gradients[r3];
            hist.entries[b3].hess_sum += hessians[r3];
            hist.entries[b3].count    += 1;
        }
    }

    // Scalar tail
    for (; i < n_rows; ++i) {
        uint32_t row = row_indices[i];
        uint8_t  bin = bin_col[row];
        hist.entries[bin].grad_sum += gradients[row];
        hist.entries[bin].hess_sum += hessians[row];
        hist.entries[bin].count    += 1;
    }
}

// NEON gradient computation: g_i = pred_i - y_i, h_i = 1.0 (squared loss)
void compute_gradients_simd(
    const double* __restrict predictions,
    const double* __restrict targets,
    double*       __restrict gradients,
    double*       __restrict hessians,
    size_t n)
{
    size_t i = 0;
    const size_t n2 = n & ~size_t(1);   // NEON: 2 doubles per register
    const float64x2_t ones = vdupq_n_f64(1.0);

    // Process 4 doubles per iteration (2 × float64x2_t)
    const size_t n4 = n & ~size_t(3);
    for (; i < n4; i += 4) {
        float64x2_t pred0 = vld1q_f64(&predictions[i]);
        float64x2_t tgt0  = vld1q_f64(&targets[i]);
        float64x2_t pred1 = vld1q_f64(&predictions[i + 2]);
        float64x2_t tgt1  = vld1q_f64(&targets[i + 2]);

        float64x2_t grad0 = vsubq_f64(pred0, tgt0);
        float64x2_t grad1 = vsubq_f64(pred1, tgt1);

        vst1q_f64(&gradients[i],     grad0);
        vst1q_f64(&gradients[i + 2], grad1);
        vst1q_f64(&hessians[i],      ones);
        vst1q_f64(&hessians[i + 2],  ones);
    }

    for (; i < n; ++i) {
        gradients[i] = predictions[i] - targets[i];
        hessians[i]  = 1.0;
    }
}

// NEON prediction update: pred[i] += lr * leaf_values[leaf_index[i]]
void update_predictions_simd(
    double* __restrict predictions,
    const double*      leaf_values,
    const uint32_t*    leaf_indices,
    double             learning_rate,
    size_t             n)
{
    const float64x2_t v_lr = vdupq_n_f64(learning_rate);
    size_t i = 0;
    const size_t n2 = n & ~size_t(1);

    for (; i < n2; i += 2) {
        float64x2_t pred = vld1q_f64(&predictions[i]);

        // Gather leaf values (scalar — indices are irregular)
        double lv[2];
        lv[0] = leaf_values[leaf_indices[i]];
        lv[1] = leaf_values[leaf_indices[i + 1]];
        float64x2_t v_lv = vld1q_f64(lv);

        // pred += lr * leaf_value  (FMA)
        pred = vfmaq_f64(pred, v_lr, v_lv);
        vst1q_f64(&predictions[i], pred);
    }

    for (; i < n; ++i)
        predictions[i] += learning_rate * leaf_values[leaf_indices[i]];
}

// NEON RMSE computation
double compute_rmse_simd(
    const double* __restrict predictions,
    const double* __restrict targets,
    size_t n)
{
    float64x2_t v_sse0 = vdupq_n_f64(0.0);
    float64x2_t v_sse1 = vdupq_n_f64(0.0);

    size_t i = 0;
    const size_t n4 = n & ~size_t(3);

    for (; i < n4; i += 4) {
        float64x2_t p0 = vld1q_f64(&predictions[i]);
        float64x2_t t0 = vld1q_f64(&targets[i]);
        float64x2_t p1 = vld1q_f64(&predictions[i + 2]);
        float64x2_t t1 = vld1q_f64(&targets[i + 2]);

        float64x2_t e0 = vsubq_f64(p0, t0);
        float64x2_t e1 = vsubq_f64(p1, t1);

        v_sse0 = vfmaq_f64(v_sse0, e0, e0);   // sse += err * err
        v_sse1 = vfmaq_f64(v_sse1, e1, e1);
    }

    double sse = hsum_f64x2(vaddq_f64(v_sse0, v_sse1));

    for (; i < n; ++i) {
        double e = predictions[i] - targets[i];
        sse += e * e;
    }
    return std::sqrt(sse / static_cast<double>(n));
}

#else
// ---- Fallback: scalar versions with same API ----

void build_histogram_simd(
    const uint8_t* bin_col, const double* gradients, const double* hessians,
    const uint32_t* row_indices, size_t n_rows, Histogram& hist) {
    build_histogram_scalar(bin_col, gradients, hessians,
                           row_indices, n_rows, hist);
}

void compute_gradients_simd(
    const double* predictions, const double* targets,
    double* gradients, double* hessians, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        gradients[i] = predictions[i] - targets[i];
        hessians[i]  = 1.0;
    }
}

double compute_rmse_simd(const double* predictions,
                          const double* targets, size_t n) {
    double sse = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = predictions[i] - targets[i];
        sse += e * e;
    }
    return std::sqrt(sse / static_cast<double>(n));
}

#endif // HAS_NEON

// ============================================================================
//  SECTION 6: DECISION TREE NODE & TREE
// ============================================================================

struct TreeNode {
    int32_t  feature_id  = -1;
    uint8_t  split_bin   = 0;
    float    split_value = 0.0f;
    double   leaf_value  = 0.0;
    int32_t  left_child  = -1;
    int32_t  right_child = -1;

    bool is_leaf() const { return feature_id == -1; }
};

struct SplitInfo {
    int32_t  feature     = -1;
    uint8_t  bin         = 0;
    double   gain        = -std::numeric_limits<double>::infinity();
    double   left_grad   = 0, left_hess  = 0;
    double   right_grad  = 0, right_hess = 0;
    uint32_t left_count  = 0, right_count= 0;

    bool valid() const { return gain > 0 && feature >= 0; }
};

class DecisionTree {
public:
    std::vector<TreeNode> nodes;
    size_t max_depth;
    size_t max_bins;
    size_t min_samples_leaf;
    double lambda;
    double gamma;

    DecisionTree(const GBConfig& cfg)
        : max_depth(cfg.max_depth)
        , max_bins(cfg.max_bins)
        , min_samples_leaf(cfg.min_samples_leaf)
        , lambda(cfg.lambda_l2)
        , gamma(cfg.gamma)
    {
        nodes.reserve(1 << (max_depth + 1));
    }

    double compute_leaf_weight(double grad_sum, double hess_sum) const {
        return -grad_sum / (hess_sum + lambda);
    }

    double compute_gain(double lg, double lh,
                        double rg, double rh) const {
        double gl = (lg * lg) / (lh + lambda);
        double gr = (rg * rg) / (rh + lambda);
        double gw = ((lg + rg) * (lg + rg)) / (lh + rh + lambda);
        return 0.5 * (gl + gr - gw) - gamma;
    }

    SplitInfo find_best_split_feature(
        const Histogram& hist, int32_t feature_id,
        double total_grad, double total_hess,
        uint32_t total_count) const
    {
        SplitInfo best;
        double cum_grad = 0, cum_hess = 0;
        uint32_t cum_count = 0;

        for (size_t b = 0; b + 1 < hist.size(); ++b) {
            cum_grad  += hist.entries[b].grad_sum;
            cum_hess  += hist.entries[b].hess_sum;
            cum_count += hist.entries[b].count;

            if (cum_count < min_samples_leaf) continue;
            uint32_t rc = total_count - cum_count;
            if (rc < min_samples_leaf) break;

            double rg = total_grad - cum_grad;
            double rh = total_hess - cum_hess;

            double gain = compute_gain(cum_grad, cum_hess, rg, rh);
            if (gain > best.gain) {
                best.gain        = gain;
                best.feature     = feature_id;
                best.bin         = static_cast<uint8_t>(b);
                best.left_grad   = cum_grad;
                best.left_hess   = cum_hess;
                best.right_grad  = rg;
                best.right_hess  = rh;
                best.left_count  = cum_count;
                best.right_count = rc;
            }
        }
        return best;
    }

    int32_t build_node(
        const ColumnBinnedData& data,
        const BinMapper&        mapper,
        const double*           gradients,
        const double*           hessians,
        std::vector<uint32_t>&  row_indices,
        size_t                  depth,
        bool                    use_simd)
    {
        size_t n = row_indices.size();

        double total_grad = 0, total_hess = 0;
        for (uint32_t r : row_indices) {
            total_grad += gradients[r];
            total_hess += hessians[r];
        }

        if (depth >= max_depth || n <= min_samples_leaf) {
            int32_t nid = static_cast<int32_t>(nodes.size());
            nodes.push_back({});
            nodes.back().leaf_value =
                compute_leaf_weight(total_grad, total_hess);
            return nid;
        }

        SplitInfo best_split;
        Histogram hist(data.max_bins);

        for (size_t f = 0; f < data.n_cols; ++f) {
            if (use_simd) {
                build_histogram_simd(
                    data.col(f), gradients, hessians,
                    row_indices.data(), n, hist);
            } else {
                build_histogram_scalar(
                    data.col(f), gradients, hessians,
                    row_indices.data(), n, hist);
            }

            SplitInfo sp = find_best_split_feature(
                hist, static_cast<int32_t>(f),
                total_grad, total_hess,
                static_cast<uint32_t>(n));

            if (sp.gain > best_split.gain)
                best_split = sp;
        }

        if (!best_split.valid()) {
            int32_t nid = static_cast<int32_t>(nodes.size());
            nodes.push_back({});
            nodes.back().leaf_value =
                compute_leaf_weight(total_grad, total_hess);
            return nid;
        }

        std::vector<uint32_t> left_rows, right_rows;
        left_rows.reserve(best_split.left_count);
        right_rows.reserve(best_split.right_count);

        const uint8_t* col = data.col(best_split.feature);
        for (uint32_t r : row_indices) {
            if (col[r] <= best_split.bin)
                left_rows.push_back(r);
            else
                right_rows.push_back(r);
        }

        int32_t nid = static_cast<int32_t>(nodes.size());
        nodes.push_back({});
        nodes[nid].feature_id  = best_split.feature;
        nodes[nid].split_bin   = best_split.bin;

        const auto& edges = mapper.bin_edges[best_split.feature];
        if (best_split.bin < edges.size())
            nodes[nid].split_value = edges[best_split.bin];

        nodes[nid].left_child = build_node(
            data, mapper, gradients, hessians,
            left_rows, depth + 1, use_simd);
        nodes[nid].right_child = build_node(
            data, mapper, gradients, hessians,
            right_rows, depth + 1, use_simd);

        return nid;
    }

    double predict_one(const float* features) const {
        int32_t idx = 0;
        while (!nodes[idx].is_leaf()) {
            float val = features[nodes[idx].feature_id];
            idx = (val <= nodes[idx].split_value)
                ? nodes[idx].left_child
                : nodes[idx].right_child;
        }
        return nodes[idx].leaf_value;
    }

    double predict_one_binned(const ColumnBinnedData& data,
                               uint32_t row) const {
        int32_t idx = 0;
        while (!nodes[idx].is_leaf()) {
            uint8_t bin = data.col(nodes[idx].feature_id)[row];
            idx = (bin <= nodes[idx].split_bin)
                ? nodes[idx].left_child
                : nodes[idx].right_child;
        }
        return nodes[idx].leaf_value;
    }
};

// ============================================================================
//  SECTION 7: GRADIENT BOOSTING ENSEMBLE
// ============================================================================

class GradientBoosting {
public:
    GBConfig                   config;
    std::vector<DecisionTree>  trees;
    double                     base_score = 0.0;
    BinMapper                  bin_mapper;

    void fit(const DenseMatrix& X, const std::vector<double>& y) {
        size_t n = X.n_rows;

        std::cout << "\n[1] Binning data (" << config.max_bins << " bins)...\n";
        {
            Timer t("Binning");
            bin_mapper.fit(X, config.max_bins);
        }

        ColumnBinnedData binned;
        {
            Timer t("Column-major transform");
            binned = create_binned_data(X, bin_mapper);
        }

        base_score = 0.0;
        for (double yi : y) base_score += yi;
        base_score /= static_cast<double>(n);

        std::vector<double> predictions(n, base_score);
        std::vector<double> gradients(n), hessians(n);

        std::vector<uint32_t> all_rows(n);
        std::iota(all_rows.begin(), all_rows.end(), 0u);

        std::cout << "[2] Training " << config.n_trees << " trees...\n";
        Timer train_timer("Total training");

        for (size_t t = 0; t < config.n_trees; ++t) {
            if (config.use_simd) {
                compute_gradients_simd(
                    predictions.data(), y.data(),
                    gradients.data(), hessians.data(), n);
            } else {
                for (size_t i = 0; i < n; ++i) {
                    gradients[i] = predictions[i] - y[i];
                    hessians[i]  = 1.0;
                }
            }

            std::vector<uint32_t> sample_rows;
            if (config.subsample < 1.0) {
                std::mt19937 rng(static_cast<uint32_t>(t));
                for (uint32_t r : all_rows)
                    if (std::uniform_real_distribution<>(0, 1)(rng)
                        < config.subsample)
                        sample_rows.push_back(r);
            } else {
                sample_rows = all_rows;
            }

            DecisionTree tree(config);
            tree.build_node(binned, bin_mapper,
                           gradients.data(), hessians.data(),
                           sample_rows, 0, config.use_simd);

            for (size_t i = 0; i < n; ++i) {
                double lv = tree.predict_one_binned(
                    binned, static_cast<uint32_t>(i));
                predictions[i] += config.learning_rate * lv;
            }

            trees.push_back(std::move(tree));

            if ((t + 1) % 10 == 0 || t == 0) {
                double rmse = compute_rmse_simd(
                    predictions.data(), y.data(), n);
                std::cout << "    Tree " << std::setw(3) << (t + 1)
                          << "/" << config.n_trees
                          << "  RMSE=" << std::fixed
                          << std::setprecision(6) << rmse << "\n";
            }
        }
    }

    std::vector<double> predict(const DenseMatrix& X) const {
        std::vector<double> preds(X.n_rows, base_score);
        for (const auto& tree : trees)
            for (size_t i = 0; i < X.n_rows; ++i)
                preds[i] += config.learning_rate *
                            tree.predict_one(X.row_ptr(i));
        return preds;
    }

    double rmse(const std::vector<double>& preds,
                const std::vector<double>& truth) const {
        double sse = 0;
        for (size_t i = 0; i < preds.size(); ++i) {
            double e = preds[i] - truth[i];
            sse += e * e;
        }
        return std::sqrt(sse / preds.size());
    }
};

// ============================================================================
//  SECTION 8: CODE GENERATION
// ============================================================================

// 8a. Template metaprogramming: compile-time unrolled tree evaluator
template<size_t MaxDepth>
struct CompiledTreeEvaluator {
    static constexpr size_t N_NODES = (1 << (MaxDepth + 1)) - 1;

    int32_t features[N_NODES];
    float   thresholds[N_NODES];
    double  leaf_values[N_NODES];
    bool    is_leaf_flag[N_NODES];

    void load(const DecisionTree& tree) {
        std::memset(features,     -1, sizeof(features));
        std::memset(is_leaf_flag,  1, sizeof(is_leaf_flag));
        std::memset(leaf_values,   0, sizeof(leaf_values));
        if (!tree.nodes.empty())
            load_node(tree, 0, 0);
    }

    void load_node(const DecisionTree& tree,
                   int32_t tree_idx, size_t flat_idx) {
        if (tree_idx < 0 ||
            tree_idx >= (int32_t)tree.nodes.size()) return;
        if (flat_idx >= N_NODES) return;

        const auto& node = tree.nodes[tree_idx];
        is_leaf_flag[flat_idx] = node.is_leaf();
        features[flat_idx]     = node.feature_id;
        thresholds[flat_idx]   = node.split_value;
        leaf_values[flat_idx]  = node.leaf_value;

        if (!node.is_leaf()) {
            load_node(tree, node.left_child,  2 * flat_idx + 1);
            load_node(tree, node.right_child, 2 * flat_idx + 2);
        }
    }

    double predict(const float* feat_in) const {
        size_t idx = 0;
        for (size_t d = 0; d < MaxDepth; ++d) {
            if (is_leaf_flag[idx]) return leaf_values[idx];
            bool go_left = (feat_in[features[idx]] <= thresholds[idx]);
            idx = go_left ? (2 * idx + 1) : (2 * idx + 2);
        }
        return leaf_values[idx];
    }
};

// 8b. Runtime C code generation
std::string generate_c_code(const GradientBoosting& model,
                             const std::string& func_name = "predict") {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(10);

    oss << "// Auto-generated gradient boosting predictor\n"
        << "// Trees: " << model.trees.size()
        << ", LR: " << model.config.learning_rate << "\n\n"
        << "#include <stddef.h>\n\n";

    for (size_t t = 0; t < model.trees.size(); ++t) {
        oss << "static double tree_" << t << "(const float* x) {\n";
        std::function<void(int32_t, int)> gen_node;
        gen_node = [&](int32_t idx, int indent) {
            const auto& node = model.trees[t].nodes[idx];
            std::string pad(indent * 2, ' ');
            if (node.is_leaf()) {
                oss << pad << "return " << node.leaf_value << ";\n";
            } else {
                oss << pad << "if (x[" << node.feature_id
                    << "] <= " << node.split_value << "f) {\n";
                gen_node(node.left_child, indent + 1);
                oss << pad << "} else {\n";
                gen_node(node.right_child, indent + 1);
                oss << pad << "}\n";
            }
        };
        gen_node(0, 1);
        oss << "}\n\n";
    }

    oss << "double " << func_name << "(const float* x) {\n"
        << "  double sum = " << model.base_score << ";\n";
    for (size_t t = 0; t < model.trees.size(); ++t)
        oss << "  sum += " << model.config.learning_rate
            << " * tree_" << t << "(x);\n";
    oss << "  return sum;\n}\n";

    return oss.str();
}

// ============================================================================
//  SECTION 9: SPARSE vs DENSE BENCHMARK
// ============================================================================

void benchmark_sparse_vs_dense(
    const DenseMatrix& X_dense,
    const std::vector<double>& y,
    const GBConfig& cfg)
{
    std::cout << "\n------------------------------------------------------------\n"
              << "  Sparse vs Dense Histogram Benchmark\n"
              << "------------------------------------------------------------\n";

    CSRMatrix X_sparse = to_csr(X_dense);
    std::cout << "  Density: " << std::fixed << std::setprecision(1)
              << (1.0 - X_sparse.sparsity) * 100.0 << "%\n";
    std::cout << "  Dense  storage: "
              << X_dense.data.size() * sizeof(float) / 1024 << " KB\n";
    std::cout << "  Sparse storage: "
              << (X_sparse.values.size() * sizeof(float) +
                  X_sparse.col_indices.size() * sizeof(uint32_t) +
                  X_sparse.row_ptr.size() * sizeof(size_t)) / 1024
              << " KB\n";

    BinMapper mapper;
    mapper.fit(X_dense, cfg.max_bins);
    auto binned = create_binned_data(X_dense, mapper);

    std::vector<double> gradients(X_dense.n_rows, 0.1);
    std::vector<double> hessians(X_dense.n_rows, 1.0);
    std::vector<uint32_t> rows(X_dense.n_rows);
    std::iota(rows.begin(), rows.end(), 0u);

    Histogram hist(cfg.max_bins);
    constexpr int ITERS = 100;

    // Dense Scalar
    {
        Timer t;
        for (int iter = 0; iter < ITERS; ++iter)
            for (size_t f = 0; f < X_dense.n_cols; ++f)
                build_histogram_scalar(
                    binned.col(f), gradients.data(), hessians.data(),
                    rows.data(), rows.size(), hist);
        std::cout << "  Dense  Scalar:  " << std::setprecision(2)
                  << t.elapsed_ms() << " ms (" << ITERS << " iters × "
                  << X_dense.n_cols << " features)\n";
    }

    // Dense SIMD (NEON)
    {
        Timer t;
        for (int iter = 0; iter < ITERS; ++iter)
            for (size_t f = 0; f < X_dense.n_cols; ++f)
                build_histogram_simd(
                    binned.col(f), gradients.data(), hessians.data(),
                    rows.data(), rows.size(), hist);
        std::cout << "  Dense  NEON:    " << std::setprecision(2)
                  << t.elapsed_ms() << " ms\n";
    }

    // Sparse CSR
    {
        Timer t;
        Histogram sparse_hist(cfg.max_bins);
        for (int iter = 0; iter < ITERS; ++iter) {
            for (size_t f = 0; f < X_dense.n_cols; ++f) {
                sparse_hist.clear();
                for (size_t r = 0; r < X_sparse.n_rows; ++r) {
                    float val = 0.0f;
                    for (size_t j = X_sparse.row_ptr[r];
                         j < X_sparse.row_ptr[r + 1]; ++j) {
                        if (X_sparse.col_indices[j] == f) {
                            val = X_sparse.values[j];
                            break;
                        }
                    }
                    uint8_t bin = mapper.map_value(f, val);
                    sparse_hist.entries[bin].grad_sum += gradients[r];
                    sparse_hist.entries[bin].hess_sum += hessians[r];
                    sparse_hist.entries[bin].count    += 1;
                }
            }
        }
        std::cout << "  Sparse (CSR):   " << std::setprecision(2)
                  << t.elapsed_ms() << " ms\n";
    }
}

// ============================================================================
//  SECTION 10: COMPILER & PLATFORM INFO
// ============================================================================

void print_compiler_info() {
    std::cout << "============================================================\n"
              << "  Compiler / Platform Information\n"
              << "============================================================\n";

#if defined(__clang__)
    std::cout << "  Compiler: Apple Clang " << __clang_major__ << "."
              << __clang_minor__ << "." << __clang_patchlevel__ << "\n";
#elif defined(__GNUC__)
    std::cout << "  Compiler: GCC " << __GNUC__ << "."
              << __GNUC_MINOR__ << "\n";
#endif

    std::cout << "  Architecture: ";
#if defined(__aarch64__)
    std::cout << "AArch64 (ARM64)";
#elif defined(__x86_64__)
    std::cout << "x86_64";
#else
    std::cout << "Unknown";
#endif
    std::cout << "\n";

    std::cout << "  SIMD: ";
#if HAS_NEON
    std::cout << "ARM NEON (128-bit, float64x2)";
#elif HAS_AVX2
    std::cout << "x86 AVX2";
#elif HAS_SSE2
    std::cout << "x86 SSE2";
#else
    std::cout << "Scalar only";
#endif
    std::cout << "\n";

#if defined(__ARM_FEATURE_FMA)
    std::cout << "  FMA: Yes (vfmaq_f64)\n";
#elif defined(__aarch64__)
    std::cout << "  FMA: Yes (AArch64 always has FMA)\n";
#endif

    std::cout << "  Optimization: ";
#if defined(__OPTIMIZE__)
    std::cout << "ON\n";
#else
    std::cout << "OFF (debug build)\n";
#endif

    std::cout << "  sizeof(HistEntry) = " << sizeof(HistEntry) << " bytes\n"
              << "  sizeof(TreeNode)  = " << sizeof(TreeNode)  << " bytes\n";
}

// ============================================================================
//  SECTION 11: SYNTHETIC DATA GENERATION (Friedman #1)
// ============================================================================

struct Dataset {
    DenseMatrix         X;
    std::vector<double> y;
};

Dataset generate_friedman1(size_t n_samples, size_t n_features,
                           double noise = 1.0, double sparsity = 0.0,
                           uint32_t seed = 42) {
    assert(n_features >= 5);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    std::normal_distribution<double> noise_dist(0.0, noise);

    Dataset ds;
    ds.X.n_rows = n_samples;
    ds.X.n_cols = n_features;
    ds.X.data.resize(n_samples * n_features);
    ds.y.resize(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            float val = unif(rng);
            if (j >= 5 && unif(rng) < sparsity) val = 0.0f;
            ds.X.get(i, j) = val;
        }

        double x0 = ds.X.get(i, 0), x1 = ds.X.get(i, 1);
        double x2 = ds.X.get(i, 2), x3 = ds.X.get(i, 3);
        double x4 = ds.X.get(i, 4);

        ds.y[i] = 10.0 * std::sin(M_PI * x0 * x1)
                 + 20.0 * (x2 - 0.5) * (x2 - 0.5)
                 + 10.0 * x3
                 + 5.0  * x4
                 + noise_dist(rng);
    }
    return ds;
}

// ============================================================================
//  SECTION 12: MAIN — FULL EXPERIMENT PIPELINE
// ============================================================================

int main() {
    print_compiler_info();

    // ------------------------------------------------------------------
    //  Experiment 1: GBM Training — SIMD (NEON) vs Scalar
    // ------------------------------------------------------------------
    std::cout << "\n============================================================\n"
              << "  EXPERIMENT 1: Gradient Boosting — NEON vs Scalar\n"
              << "============================================================\n";

    constexpr size_t N_TRAIN    = 50000;
    constexpr size_t N_TEST     = 10000;
    constexpr size_t N_FEATURES = 20;

    auto train_data = generate_friedman1(N_TRAIN, N_FEATURES, 1.0, 0.0, 42);
    auto test_data  = generate_friedman1(N_TEST,  N_FEATURES, 1.0, 0.0, 123);

    std::cout << "  Train: " << N_TRAIN << " × " << N_FEATURES << "\n"
              << "  Test:  " << N_TEST  << " × " << N_FEATURES << "\n";

    // SIMD path
    double simd_train_ms, simd_rmse_val;
    {
        std::cout << "\n--- NEON SIMD Path ---\n";
        GradientBoosting gb;
        gb.config.n_trees       = 50;
        gb.config.max_depth     = 6;
        gb.config.max_bins      = 256;
        gb.config.learning_rate = 0.1;
        gb.config.lambda_l2     = 1.0;
        gb.config.use_simd      = true;

        Timer t;
        gb.fit(train_data.X, train_data.y);
        simd_train_ms = t.elapsed_ms();

        auto preds = gb.predict(test_data.X);
        simd_rmse_val = gb.rmse(preds, test_data.y);
        std::cout << "  Test RMSE: " << simd_rmse_val << "\n";

        // Code generation
        std::cout << "\n--- Generated C Code (first 50 lines) ---\n";
        std::string code = generate_c_code(gb, "gb_predict");
        std::istringstream iss(code);
        std::string line;
        int lc = 0;
        while (std::getline(iss, line) && lc < 50) {
            std::cout << "  " << line << "\n";
            ++lc;
        }
        size_t total_lines = std::count(
            code.begin(), code.end(), '\n');
        if (lc == 50)
            std::cout << "  ... (" << total_lines
                      << " total lines)\n";

        std::ofstream out("generated_predictor.c");
        out << code;
        std::cout << "\n  Saved to: generated_predictor.c\n";
    }

    // Scalar path
    double scalar_train_ms, scalar_rmse_val;
    {
        std::cout << "\n--- Scalar Path ---\n";
        GradientBoosting gb;
        gb.config.n_trees       = 50;
        gb.config.max_depth     = 6;
        gb.config.max_bins      = 256;
        gb.config.learning_rate = 0.1;
        gb.config.lambda_l2     = 1.0;
        gb.config.use_simd      = false;

        Timer t;
        gb.fit(train_data.X, train_data.y);
        scalar_train_ms = t.elapsed_ms();

        auto preds = gb.predict(test_data.X);
        scalar_rmse_val = gb.rmse(preds, test_data.y);
        std::cout << "  Test RMSE: " << scalar_rmse_val << "\n";
    }

    std::cout << "\n--- Experiment 1 Summary ---\n"
              << "  NEON   training: " << std::setprecision(1)
              << simd_train_ms << " ms  |  RMSE: "
              << std::setprecision(6) << simd_rmse_val << "\n"
              << "  Scalar training: " << std::setprecision(1)
              << scalar_train_ms << " ms  |  RMSE: "
              << std::setprecision(6) << scalar_rmse_val << "\n"
              << "  Speedup: " << std::setprecision(2)
              << scalar_train_ms / simd_train_ms << "×\n";

    // ------------------------------------------------------------------
    //  Experiment 2: Sparse vs Dense at various sparsity levels
    // ------------------------------------------------------------------
    std::cout << "\n============================================================\n"
              << "  EXPERIMENT 2: Sparse vs Dense Feature Handling\n"
              << "============================================================\n";

    for (double sparsity : {0.0, 0.5, 0.8, 0.95}) {
        std::cout << "\n--- Sparsity = "
                  << static_cast<int>(sparsity * 100) << "% ---\n";
        auto sparse_data = generate_friedman1(
            10000, N_FEATURES, 1.0, sparsity, 42);

        GBConfig cfg;
        cfg.max_bins = 64;
        benchmark_sparse_vs_dense(sparse_data.X, sparse_data.y, cfg);
    }

    // ------------------------------------------------------------------
    //  Experiment 3: Template-unrolled tree evaluator
    // ------------------------------------------------------------------
    std::cout << "\n============================================================\n"
              << "  EXPERIMENT 3: Compiled Tree Evaluator (Templates)\n"
              << "============================================================\n";
    {
        GradientBoosting gb;
        gb.config.n_trees   = 10;
        gb.config.max_depth = 4;
        gb.config.max_bins  = 64;
        gb.config.use_simd  = true;
        gb.fit(train_data.X, train_data.y);

        CompiledTreeEvaluator<4> compiled;
        compiled.load(gb.trees[0]);

        constexpr int EVAL_ITERS = 1000;
        double regular_ms, compiled_ms;

        {
            Timer t;
            volatile double sink = 0;
            for (int iter = 0; iter < EVAL_ITERS; ++iter)
                for (size_t i = 0; i < test_data.X.n_rows; ++i)
                    sink = gb.trees[0].predict_one(
                        test_data.X.row_ptr(i));
            regular_ms = t.elapsed_ms();
            std::cout << "  Regular tree eval:  "
                      << std::setprecision(2) << regular_ms << " ms\n";
        }
        {
            Timer t;
            volatile double sink = 0;
            for (int iter = 0; iter < EVAL_ITERS; ++iter)
                for (size_t i = 0; i < test_data.X.n_rows; ++i)
                    sink = compiled.predict(
                        test_data.X.row_ptr(i));
            compiled_ms = t.elapsed_ms();
            std::cout << "  Compiled tree eval: "
                      << std::setprecision(2) << compiled_ms << " ms\n";
        }

        std::cout << "  Speedup: " << std::setprecision(2)
                  << regular_ms / compiled_ms << "×\n";

        bool correct = true;
        for (size_t i = 0; i < std::min(test_data.X.n_rows,
                                         size_t(100)); ++i) {
            double a = gb.trees[0].predict_one(
                test_data.X.row_ptr(i));
            double b = compiled.predict(test_data.X.row_ptr(i));
            if (std::abs(a - b) > 1e-10) { correct = false; break; }
        }
        std::cout << "  Correctness: "
                  << (correct ? "PASS ✓" : "FAIL ✗") << "\n";
    }

    // ------------------------------------------------------------------
    //  Experiment 4: Compiler flag ablation guide
    // ------------------------------------------------------------------
    std::cout << "\n============================================================\n"
              << "  EXPERIMENT 4: Compiler Flag Ablation Guide\n"
              << "============================================================\n";
    std::cout << R"(
  Re-compile with each flag set and compare timings:

  | Flag                | What it does                              |
  |---------------------|-------------------------------------------|
  | -O0                 | No optimization (baseline)                |
  | -O2                 | Standard optimization                     |
  | -O3                 | Aggressive (vectorization, inline)         |
  | -Os                 | Size optimization                         |
  | -mcpu=apple-m1      | Apple M1 tuning                           |
  | -mcpu=apple-m2      | Apple M2 tuning                           |
  | -mcpu=native        | Auto-detect (may not work on all clang)   |
  | -flto               | Link-time optimization                    |
  | -fprofile-generate   | PGO: generate profile data                |
  | -fprofile-use        | PGO: use profile for optimization         |

  Ablation script for macOS:

    for OPT in "-O0" "-O1" "-O2" "-O3" "-O3 -mcpu=apple-m1" \
               "-O3 -mcpu=apple-m1 -flto"; do
      echo "=== $OPT ==="
      clang++ -std=c++17 $OPT -o gboost gboost_neon.cpp 2>/dev/null
      ./gboost 2>&1 | grep -E "TIMER|Speedup|RMSE"
      echo ""
    done
)" << "\n";

    // ------------------------------------------------------------------
    //  Experiment 5: Histogram bin count vs accuracy/speed
    // ------------------------------------------------------------------
    std::cout << "============================================================\n"
              << "  EXPERIMENT 5: Histogram Bin Count vs Accuracy/Speed\n"
              << "============================================================\n\n";

    std::cout << "  " << std::left
              << std::setw(6)  << "Bins"
              << std::setw(14) << "Train (ms)"
              << std::setw(14) << "RMSE"
              << std::setw(14) << "Infer (ms)" << "\n";
    std::cout << "  " << std::string(48, '-') << "\n";

    for (size_t bins : {16, 32, 64, 128, 256}) {
        GradientBoosting gb;
        gb.config.n_trees   = 30;
        gb.config.max_depth = 6;
        gb.config.max_bins  = bins;
        gb.config.use_simd  = true;

        Timer t;
        gb.fit(train_data.X, train_data.y);
        double train_ms = t.elapsed_ms();

        Timer t2;
        auto preds = gb.predict(test_data.X);
        double infer_ms = t2.elapsed_ms();

        double test_rmse = gb.rmse(preds, test_data.y);

        std::cout << "  " << std::left
                  << std::setw(6)  << bins
                  << std::setw(14) << std::setprecision(1)
                  << std::fixed    << train_ms
                  << std::setw(14) << std::setprecision(6) << test_rmse
                  << std::setw(14) << std::setprecision(2) << infer_ms
                  << "\n";
    }

    // ------------------------------------------------------------------
    //  Summary
    // ------------------------------------------------------------------
    std::cout << "\n============================================================\n"
              << "  All experiments complete.\n"
              << "============================================================\n\n";

    std::cout << "  Output files:\n"
              << "    generated_predictor.c  — compilable C inference code\n\n"
              << "  Next steps:\n"
              << "    1. Run ablation script (Experiment 4)\n"
              << "    2. Profile with Instruments:\n"
              << "       xcrun xctrace record --template 'Time Profiler'"
              << " --launch ./gboost\n"
              << "    3. Compare against XGBoost Python:\n"
              << "       pip install xgboost && python compare_xgb.py\n\n";

    return 0;
}
