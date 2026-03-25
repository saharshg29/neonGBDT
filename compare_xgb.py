#!/usr/bin/env python3
"""
compare_xgb.py — Compare our C++ GBM against XGBoost on the same dataset.

Usage:
    pip install xgboost numpy scikit-learn matplotlib
    python compare_xgb.py
"""

import time
import subprocess
import json
import numpy as np

# ============================================================================
#  Generate the SAME Friedman #1 dataset as our C++ code
# ============================================================================
def generate_friedman1(n_samples, n_features, noise=1.0, seed=42):
    """Matches the C++ generate_friedman1() exactly."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_samples, n_features)).astype(np.float32)

    y = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20.0 * (X[:, 2] - 0.5) ** 2
         + 10.0 * X[:, 3]
         + 5.0  * X[:, 4]
         + rng.normal(0, noise, n_samples))

    return X, y


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ============================================================================
#  Run experiments
# ============================================================================
def main():
    print("=" * 70)
    print("  XGBoost vs Custom C++ Gradient Boosting Comparison")
    print("=" * 70)

    N_TRAIN    = 50000
    N_TEST     = 10000
    N_FEATURES = 20

    # Generate data (same seeds as C++ code)
    print(f"\n[1] Generating data: train={N_TRAIN}, test={N_TEST}, "
          f"features={N_FEATURES}")
    X_train, y_train = generate_friedman1(N_TRAIN, N_FEATURES, 
                                           noise=1.0, seed=42)
    X_test,  y_test  = generate_friedman1(N_TEST,  N_FEATURES, 
                                           noise=1.0, seed=123)

    results = {}

    # ------------------------------------------------------------------
    #  XGBoost
    # ------------------------------------------------------------------
    try:
        import xgboost as xgb
        print(f"\n[2] XGBoost version: {xgb.__version__}")

        # Match our C++ hyperparameters exactly
        configs = {
            "XGBoost (hist, 50 trees)": {
                "n_estimators": 50,
                "max_depth": 6,
                "learning_rate": 0.1,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "max_bin": 256,
                "n_jobs": 1,           # single-thread for fair comparison
                "verbosity": 0,
            },
            "XGBoost (hist, 50 trees, 4 threads)": {
                "n_estimators": 50,
                "max_depth": 6,
                "learning_rate": 0.1,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "max_bin": 256,
                "n_jobs": 4,
                "verbosity": 0,
            },
            "XGBoost (hist, 100 trees)": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "max_bin": 256,
                "n_jobs": 1,
                "verbosity": 0,
            },
        }

        for name, params in configs.items():
            print(f"\n--- {name} ---")
            model = xgb.XGBRegressor(**params)

            # Training time
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = (time.perf_counter() - t0) * 1000  # ms

            # Inference time
            t0 = time.perf_counter()
            preds = model.predict(X_test)
            infer_time = (time.perf_counter() - t0) * 1000

            test_rmse = rmse(y_test, preds)

            print(f"  Train:  {train_time:.1f} ms")
            print(f"  Infer:  {infer_time:.1f} ms")
            print(f"  RMSE:   {test_rmse:.6f}")

            results[name] = {
                "train_ms": train_time,
                "infer_ms": infer_time,
                "rmse": test_rmse,
            }

    except ImportError:
        print("\n[!] XGBoost not installed. Run: pip install xgboost")
        print("    Skipping XGBoost benchmarks.\n")

    # ------------------------------------------------------------------
    #  Run our C++ implementation
    # ------------------------------------------------------------------
    print("\n[3] Running C++ implementation...")

    cpp_results = {}
    try:
        # Compile
        print("  Compiling...")
        ret = subprocess.run(
            ["clang++", "-std=c++17", "-O3", "-mcpu=apple-m1",
             "-o", "gboost_bench", "gboost_neon.cpp"],
            capture_output=True, text=True
        )
        if ret.returncode != 0:
            print(f"  Compile error: {ret.stderr}")
            raise RuntimeError("Compilation failed")

        # Run
        print("  Running...")
        t0 = time.perf_counter()
        ret = subprocess.run(
            ["./gboost_bench"],
            capture_output=True, text=True, timeout=300
        )
        total_time = (time.perf_counter() - t0) * 1000

        output = ret.stdout
        print(f"  Total runtime: {total_time:.0f} ms")

        # Parse timers from output
        for line in output.split('\n'):
            if 'TIMER' in line and 'Total training' in line:
                ms = float(line.split(':')[-1].strip().replace(' ms', ''))
                cpp_results["train_ms"] = ms
            if 'Test RMSE' in line and 'NEON' not in line:
                try:
                    val = float(line.split(':')[-1].strip())
                    cpp_results["rmse"] = val
                except ValueError:
                    pass

        # Try to get NEON-specific results
        in_neon_section = False
        for line in output.split('\n'):
            if 'NEON SIMD Path' in line:
                in_neon_section = True
            if in_neon_section and 'TIMER' in line and 'Total training' in line:
                ms = float(line.split(':')[-1].strip().replace(' ms', ''))
                cpp_results["train_ms"] = ms
                in_neon_section = False
            if in_neon_section and 'Test RMSE' in line:
                try:
                    val = float(line.split(':')[-1].strip())
                    cpp_results["rmse"] = val
                except ValueError:
                    pass

        if cpp_results:
            results["C++ NEON (50 trees, 1 thread)"] = cpp_results

    except FileNotFoundError:
        print("  gboost_neon.cpp not found in current directory")
    except subprocess.TimeoutExpired:
        print("  C++ benchmark timed out (>300s)")
    except Exception as e:
        print(f"  Error: {e}")

    # ------------------------------------------------------------------
    #  Comparison Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  {'Implementation':<42} {'Train(ms)':>10} {'RMSE':>12}")
    print(f"  {'-'*42} {'-'*10} {'-'*12}")

    for name, r in results.items():
        train = r.get('train_ms', 0)
        r_val = r.get('rmse', 0)
        print(f"  {name:<42} {train:>9.1f}  {r_val:>11.6f}")

    # ------------------------------------------------------------------
    #  Bin count experiment (matching C++ Experiment 5)
    # ------------------------------------------------------------------
    try:
        import xgboost as xgb

        print(f"\n{'=' * 70}")
        print("  BIN COUNT ABLATION: XGBoost")
        print(f"{'=' * 70}")
        print(f"\n  {'Bins':>6} {'Train(ms)':>12} {'RMSE':>12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12}")

        for bins in [16, 32, 64, 128, 256]:
            model = xgb.XGBRegressor(
                n_estimators=30, max_depth=6, learning_rate=0.1,
                reg_lambda=1.0, tree_method="hist", max_bin=bins,
                n_jobs=1, verbosity=0
            )
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_ms = (time.perf_counter() - t0) * 1000

            preds = model.predict(X_test)
            test_rmse = rmse(y_test, preds)

            print(f"  {bins:>6} {train_ms:>11.1f}  {test_rmse:>11.6f}")

    except ImportError:
        pass

    # ------------------------------------------------------------------
    #  Generate plots if matplotlib available
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')    # non-interactive backend
        import matplotlib.pyplot as plt

        if len(results) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            names = list(results.keys())
            trains = [results[n].get('train_ms', 0) for n in names]
            rmses  = [results[n].get('rmse', 0) for n in names]

            # Short labels for readability
            short_names = []
            for n in names:
                if 'C++' in n:
                    short_names.append('C++ NEON')
                elif '4 threads' in n:
                    short_names.append('XGB 4T')
                elif '100' in n:
                    short_names.append('XGB 100T')
                else:
                    short_names.append('XGB 1T')

            colors = ['#2ecc71' if 'C++' in n else '#3498db' 
                      for n in names]

            # Training time
            axes[0].barh(short_names, trains, color=colors)
            axes[0].set_xlabel('Training Time (ms)')
            axes[0].set_title('Training Time Comparison')
            for i, v in enumerate(trains):
                axes[0].text(v + max(trains)*0.02, i, f'{v:.0f}ms',
                           va='center', fontsize=10)

            # RMSE
            axes[1].barh(short_names, rmses, color=colors)
            axes[1].set_xlabel('Test RMSE')
            axes[1].set_title('Accuracy Comparison (lower = better)')
            for i, v in enumerate(rmses):
                axes[1].text(v + max(rmses)*0.001, i, f'{v:.4f}',
                           va='center', fontsize=10)

            plt.tight_layout()
            plt.savefig('comparison_plot.png', dpi=150, bbox_inches='tight')
            print(f"\n  Plot saved to: comparison_plot.png")

    except ImportError:
        print("\n  (Install matplotlib for plots: pip install matplotlib)")

    # ------------------------------------------------------------------
    #  Save results as JSON
    # ------------------------------------------------------------------
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: comparison_results.json")

    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()