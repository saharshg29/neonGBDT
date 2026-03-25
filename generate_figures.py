#!/usr/bin/env python3
"""
generate_figures.py — Create publication-quality plots from actual results.
"""

import json
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Run: pip3 install matplotlib numpy")
    sys.exit(1)

# ---- Load results ----
results_file = 'results/parsed_results.json'
if not os.path.exists(results_file):
    print(f"ERROR: {results_file} not found!")
    print("Run first: python3 collect_results.py")
    sys.exit(1)

with open(results_file, 'r') as f:
    data = json.load(f)

os.makedirs('figures', exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

print("Generating figures from actual experiment data...\n")

# ================================================================
#  Figure 1: SIMD vs Scalar Training Time
# ================================================================
s = data['simd_vs_scalar']
fig, ax = plt.subplots(figsize=(6, 4))

values = [s['scalar_time'], s['neon_time']]
labels = ['Scalar', 'ARM NEON']
colors = ['#e74c3c', '#2ecc71']

bars = ax.bar(labels, values, color=colors, width=0.5,
              edgecolor='black', linewidth=0.8)
ax.set_ylabel('Training Time (ms)')
ax.set_title('Experiment 1: SIMD vs Scalar Training\n'
             '(50 trees, 50K samples, 20 features)')

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f'{val:.1f} ms', ha='center', va='bottom',
            fontweight='bold', fontsize=11)

speedup = s['speedup']
note_color = '#e74c3c' if speedup < 1.0 else '#27ae60'
note_text = f"Ratio: {speedup:.2f}×"
if speedup < 1.0:
    note_text += "\n(NEON slower — scatter-bound)"
ax.text(0.5, 0.85, note_text,
        transform=ax.transAxes, ha='center', fontsize=11,
        color=note_color, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor=note_color, alpha=0.9))

plt.savefig('figures/fig1_simd_vs_scalar.png')
plt.savefig('figures/fig1_simd_vs_scalar.pdf')
print("  ✓ fig1_simd_vs_scalar")
plt.close()

# ================================================================
#  Figure 2: Sparse vs Dense (grouped bars)
# ================================================================
sp = data['sparse_vs_dense']
fig, ax = plt.subplots(figsize=(8, 5))

sparsities = [r['sparsity'] for r in sp]
d_scalar = [r['dense_scalar_ms'] for r in sp]
d_neon = [r['dense_neon_ms'] for r in sp]
s_csr = [r['sparse_csr_ms'] for r in sp]

x = np.arange(len(sparsities))
w = 0.25

ax.bar(x - w, d_scalar, w, label='Dense Scalar',
       color='#3498db', edgecolor='black', linewidth=0.5)
ax.bar(x, d_neon, w, label='Dense NEON',
       color='#2ecc71', edgecolor='black', linewidth=0.5)
ax.bar(x + w, s_csr, w, label='Sparse CSR',
       color='#e74c3c', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Sparsity Level')
ax.set_ylabel('Histogram Construction Time (ms)')
ax.set_title('Experiment 2: Dense vs Sparse Across Sparsity Levels\n'
             '(10K samples, 20 features, 100 iterations)')
ax.set_xticks(x)
ax.set_xticklabels([f'{s}%' for s in sparsities])
ax.legend(loc='upper left')

plt.savefig('figures/fig2_sparse_vs_dense.png')
plt.savefig('figures/fig2_sparse_vs_dense.pdf')
print("  ✓ fig2_sparse_vs_dense")
plt.close()

# ================================================================
#  Figure 3: NEON speedup over scalar across sparsity
# ================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

neon_speedup = [d_scalar[i] / d_neon[i] for i in range(len(sp))]

ax.plot(sparsities, neon_speedup, 'o-', color='#2ecc71',
        linewidth=2.5, markersize=10)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
           label='No speedup (1.0×)')

for sp_val, su in zip(sparsities, neon_speedup):
    ax.annotate(f'{su:.2f}×', (sp_val, su),
                textcoords="offset points", xytext=(0, 12),
                ha='center', fontweight='bold', fontsize=10)

ax.set_xlabel('Data Sparsity (%)')
ax.set_ylabel('Speedup (Scalar ÷ NEON)')
ax.set_title('NEON Speedup for Dense Histogram Construction')
ax.legend()
ax.set_ylim(0, max(neon_speedup) * 1.4)

plt.savefig('figures/fig3_neon_speedup_by_sparsity.png')
plt.savefig('figures/fig3_neon_speedup_by_sparsity.pdf')
print("  ✓ fig3_neon_speedup_by_sparsity")
plt.close()

# ================================================================
#  Figure 4: Compiled Tree Evaluator
# ================================================================
c = data['compiled_eval']
fig, ax = plt.subplots(figsize=(5.5, 4))

values = [c['regular_ms'], c['compiled_ms']]
labels = ['Standard\n(pointer-chasing)', 'Template\nCompiled']
colors = ['#e74c3c', '#2ecc71']

bars = ax.bar(labels, values, color=colors, width=0.5,
              edgecolor='black', linewidth=0.8)
ax.set_ylabel('Evaluation Time (ms)')
ax.set_title('Experiment 3: Tree Evaluation Methods\n'
             '(1000 iter × 10K samples, depth 4)')

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f'{val:.1f} ms', ha='center', fontweight='bold')

ax.text(0.5, 0.85, f"Speedup: {c['speedup']:.2f}×",
        transform=ax.transAxes, ha='center', fontsize=13,
        color='#27ae60', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                  edgecolor='#27ae60', alpha=0.9))

plt.savefig('figures/fig4_compiled_eval.png')
plt.savefig('figures/fig4_compiled_eval.pdf')
print("  ✓ fig4_compiled_eval")
plt.close()

# ================================================================
#  Figure 5: Bin Count Ablation (dual y-axis)
# ================================================================
bins_data = data.get('bin_ablation', [])
if bins_data:
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    bins_list = [b['bins'] for b in bins_data]
    times = [b['train_ms'] for b in bins_data]
    rmses = [b['rmse'] for b in bins_data]

    c1, c2 = '#3498db', '#e74c3c'

    ax1.set_xlabel('Number of Histogram Bins')
    ax1.set_ylabel('Training Time (ms)', color=c1)
    l1 = ax1.plot(bins_list, times, 'o-', color=c1, linewidth=2,
                  markersize=8, label='Train Time')
    ax1.tick_params(axis='y', labelcolor=c1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test RMSE', color=c2)
    l2 = ax2.plot(bins_list, rmses, 's--', color=c2, linewidth=2,
                  markersize=8, label='RMSE')
    ax2.tick_params(axis='y', labelcolor=c2)

    lines = l1 + l2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='upper right')
    ax1.set_title('Experiment 5: Bin Count vs Speed/Accuracy')

    plt.savefig('figures/fig5_bin_ablation.png')
    plt.savefig('figures/fig5_bin_ablation.pdf')
    print("  ✓ fig5_bin_ablation")
    plt.close()

# ================================================================
#  Figure 6: Training Convergence
# ================================================================
conv = data.get('convergence', [])
if conv:
    fig, ax = plt.subplots(figsize=(7, 4))

    tree_nums = [c['tree'] for c in conv]
    rmse_vals = [c['rmse'] for c in conv]

    ax.plot(tree_nums, rmse_vals, 'o-', color='#2ecc71',
            linewidth=2, markersize=6)
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('Training RMSE')
    ax.set_title('Training Convergence (Friedman #1)')

    plt.savefig('figures/fig6_convergence.png')
    plt.savefig('figures/fig6_convergence.pdf')
    print("  ✓ fig6_convergence")
    plt.close()

# ================================================================
#  Summary
# ================================================================
print(f"\n{'='*50}")
print(f"  All figures saved to figures/")
print(f"{'='*50}")
print(f"\n  Your key numbers:")
print(f"    SIMD vs Scalar:     {data['simd_vs_scalar']['speedup']}×")
sparse_entry = data['sparse_vs_dense'][3]
bin_entry = data['bin_ablation'][3]

neon_speedup = sparse_entry['dense_scalar_ms'] / sparse_entry['dense_neon_ms']
compiled_speedup = data['compiled_eval']['speedup']
rmse = bin_entry['rmse']

print(f"NEON at 95% sparse: {neon_speedup:.2f}×")
print(f"Compiled eval:      {compiled_speedup:.2f}×")
print(f"Best bin count:     128 bins (RMSE={rmse:.6f})")