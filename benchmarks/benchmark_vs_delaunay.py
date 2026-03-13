"""
================================================================================
DEFINITIVE BENCHMARK: Fast Simplex 2D vs Scipy Delaunay
================================================================================

Exhaustive tests to determine if the Fast Simplex 2D engine can compete
with Delaunay in:
- Construction speed
- Inference speed
- Precision
- Scalability

EDA Team: Gemini · Claude · Alex
License: MIT
Version: 1.0
Trans.: GLM
================================================================================
"""

import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree
from typing import Tuple, List
import sys

# Import the Fast Simplex engine (assuming it is in the same directory)
sys.path.insert(0, '/mnt/user-data/outputs')
from fast_simplex_2d import FastSimplex2D


class DelaunayInterpolator:
    """
    Wrapper for Delaunay with barycentric interpolation
    (for fair comparison with Fast Simplex)
    """
    def __init__(self):
        self.tri = None
        self.values = None
        self.points = None
        
    def fit(self, data: np.ndarray):
        """Build Delaunay triangulation"""
        self.points = data[:, :2]
        self.values = data[:, 2]
        self.tri = Delaunay(self.points)
        return self
    
    def predict(self, point: np.ndarray) -> float:
        """
        Predict using barycentric interpolation on Delaunay simplex
        """
        # Find simplex that contains the point
        simplex_idx = self.tri.find_simplex(point)
        
        if simplex_idx == -1:
            # Point outside the convex hull
            return None
        
        # Get simplex vertices
        simplex_vertices = self.tri.simplices[simplex_idx]
        simplex_coords = self.points[simplex_vertices]
        simplex_values = self.values[simplex_vertices]
        
        # Calculate barycentric coordinates
        V1, V2, V3 = simplex_coords
        T = np.column_stack([V2 - V1, V3 - V1])
        
        try:
            w = np.linalg.solve(T, point - V1)
            w1 = 1.0 - w[0] - w[1]
            weights = np.array([w1, w[0], w[1]])
            
            # Prediction
            return float(np.dot(weights, simplex_values))
        except:
            return None


def generate_dataset(N: int, func_type: str = 'linear') -> np.ndarray:
    """
    Generate synthetic dataset
    
    Args:
        N: Number of points
        func_type: 'linear', 'quadratic', 'sinusoidal'
    
    Returns:
        Array (N, 3) with [x, y, z]
    """
    np.random.seed(42)
    x = np.random.rand(N) * 10
    y = np.random.rand(N) * 10
    
    if func_type == 'linear':
        z = x + y
    elif func_type == 'quadratic':
        z = x**2 + y**2
    elif func_type == 'sinusoidal':
        z = np.sin(x) + np.cos(y)
    else:
        raise ValueError(f"unknown func_type: {func_type}")
    
    return np.column_stack([x, y, z])


def benchmark_construction(dataset_sizes: List[int]) -> dict:
    """
    Compare construction time (fit)
    """
    print("\n" + "="*80)
    print("BENCHMARK 1: CONSTRUCTION TIME (FIT)")
    print("="*80)
    
    results = {
        'sizes': dataset_sizes,
        'fastsimplex_times': [],
        'delaunay_times': []
    }
    
    print(f"\n{'N Points':<12} {'Fast Simplex (ms)':<15} {'Delaunay (ms)':<15} {'Speedup':<12}")
    print("-" * 70)
    
    for N in dataset_sizes:
        data = generate_dataset(N, 'linear')
        
        # Fast Simplex
        start = time.perf_counter()
        fastsimplex = FastSimplex2D(max_radius=0, lambda_factor=9)
        fastsimplex.fit(data)
        time_fastsimplex = (time.perf_counter() - start) * 1000
        
        # Delaunay
        start = time.perf_counter()
        delaunay = DelaunayInterpolator()
        delaunay.fit(data)
        time_delaunay = (time.perf_counter() - start) * 1000
        
        speedup = time_delaunay / time_fastsimplex
        
        results['fastsimplex_times'].append(time_fastsimplex)
        results['delaunay_times'].append(time_delaunay)
        
        print(f"{N:<12} {time_fastsimplex:<15.2f} {time_delaunay:<15.2f} {speedup:<12.2f}x")
    
    return results


def benchmark_inference_single(N: int, n_queries: int = 1000) -> dict:
    """
    Compare inference time (predict) with individual queries
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK 2: SINGLE QUERY INFERENCE (N={N}, {n_queries} queries)")
    print("="*80)
    
    data = generate_dataset(N, 'linear')
    
    # Build both models
    fastsimplex = FastSimplex2D(max_radius=0, lambda_factor=9)
    fastsimplex.fit(data)
    
    delaunay = DelaunayInterpolator()
    delaunay.fit(data)
    
    # Generate query points
    np.random.seed(123)
    queries = np.random.rand(n_queries, 2) * 10
    
    # Fast Simplex
    start = time.perf_counter()
    for q in queries:
        fastsimplex.predict(q)
    time_fastsimplex = (time.perf_counter() - start) * 1000
    
    # Delaunay
    start = time.perf_counter()
    for q in queries:
        delaunay.predict(q)
    time_delaunay = (time.perf_counter() - start) * 1000
    
    speedup = time_delaunay / time_fastsimplex
    throughput_fastsimplex = n_queries / (time_fastsimplex / 1000)
    throughput_delaunay = n_queries / (time_delaunay / 1000)
    
    print(f"\nResults:")
    print(f"  Fast Simplex:     {time_fastsimplex:.2f} ms total | {throughput_fastsimplex:.0f} pred/s")
    print(f"  Delaunay:  {time_delaunay:.2f} ms total | {throughput_delaunay:.0f} pred/s")
    print(f"  Speedup:   {speedup:.2f}x {'(Fast Simplex WINS)' if speedup > 1 else '(Delaunay WINS)'}")
    
    return {
        'N': N,
        'n_queries': n_queries,
        'fastsimplex_time': time_fastsimplex,
        'delaunay_time': time_delaunay,
        'speedup': speedup,
        'fastsimplex_throughput': throughput_fastsimplex,
        'delaunay_throughput': throughput_delaunay
    }


def benchmark_precision(N: int, n_queries: int = 100) -> dict:
    """
    Compare precision on known function
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK 3: PRECISION (N={N}, {n_queries} queries)")
    print("="*80)
    
    data = generate_dataset(N, 'linear')
    
    # Build both models
    fastsimplex = FastSimplex2D(max_radius=0, lambda_factor=9)
    fastsimplex.fit(data)
    
    delaunay = DelaunayInterpolator()
    delaunay.fit(data)
    
    # Generate query points with known true values
    np.random.seed(456)
    queries = np.random.rand(n_queries, 2) * 10
    true_values = queries[:, 0] + queries[:, 1]  # z = x + y
    
    # Predictions
    preds_fastsimplex = []
    preds_delaunay = []
    
    for q in queries:
        pred_l = fastsimplex.predict(q)
        pred_d = delaunay.predict(q)
        preds_fastsimplex.append(pred_l if pred_l is not None else np.nan)
        preds_delaunay.append(pred_d if pred_d is not None else np.nan)
    
    preds_fastsimplex = np.array(preds_fastsimplex)
    preds_delaunay = np.array(preds_delaunay)
    
    # Filter Nones
    valid_fastsimplex = ~np.isnan(preds_fastsimplex)
    valid_delaunay = ~np.isnan(preds_delaunay)
    
    # Calculate errors
    mse_fastsimplex = np.mean((preds_fastsimplex[valid_fastsimplex] - true_values[valid_fastsimplex])**2)
    mse_delaunay = np.mean((preds_delaunay[valid_delaunay] - true_values[valid_delaunay])**2)
    
    rmse_fastsimplex = np.sqrt(mse_fastsimplex)
    rmse_delaunay = np.sqrt(mse_delaunay)
    
    success_rate_fastsimplex = np.sum(valid_fastsimplex) / len(queries) * 100
    success_rate_delaunay = np.sum(valid_delaunay) / len(queries) * 100
    
    print(f"\nResults:")
    print(f"  Fast Simplex:")
    print(f"    - RMSE: {rmse_fastsimplex:.6f}")
    print(f"    - Success rate: {success_rate_fastsimplex:.1f}%")
    print(f"  Delaunay:")
    print(f"    - RMSE: {rmse_delaunay:.6f}")
    print(f"    - Success rate: {success_rate_delaunay:.1f}%")
    
    return {
        'N': N,
        'rmse_fastsimplex': rmse_fastsimplex,
        'rmse_delaunay': rmse_delaunay,
        'success_fastsimplex': success_rate_fastsimplex,
        'success_delaunay': success_rate_delaunay
    }


def benchmark_scalability() -> dict:
    """
    Test scalability with very large datasets
    """
    print(f"\n{'='*80}")
    print("BENCHMARK 4: SCALABILITY (LARGE DATASETS)")
    print("="*80)
    
    sizes = [1000, 5000, 10000, 50000, 100000]
    n_queries = 100  # Fewer queries to not take too long
    
    results = {
        'sizes': sizes,
        'fastsimplex_fit': [],
        'delaunay_fit': [],
        'fastsimplex_predict': [],
        'delaunay_predict': []
    }
    
    print(f"\n{'N':<10} {'Fast Simplex Fit':<15} {'Del Fit':<15} {'Fast Simplex Pred':<15} {'Del Pred':<15}")
    print("-" * 80)
    
    for N in sizes:
        print(f"{N:<10}", end=" ", flush=True)
        
        data = generate_dataset(N, 'linear')
        queries = np.random.rand(n_queries, 2) * 10
        
        # Fast Simplex FIT
        start = time.perf_counter()
        fastsimplex = FastSimplex2D(max_radius=0, lambda_factor=9)
        fastsimplex.fit(data)
        t_fastsimplex_fit = (time.perf_counter() - start) * 1000
        
        # Delaunay FIT
        start = time.perf_counter()
        delaunay = DelaunayInterpolator()
        delaunay.fit(data)
        t_delaunay_fit = (time.perf_counter() - start) * 1000
        
        # Fast Simplex PREDICT
        start = time.perf_counter()
        for q in queries:
            fastsimplex.predict(q)
        t_fastsimplex_pred = (time.perf_counter() - start) * 1000
        
        # Delaunay PREDICT
        start = time.perf_counter()
        for q in queries:
            delaunay.predict(q)
        t_delaunay_pred = (time.perf_counter() - start) * 1000
        
        results['fastsimplex_fit'].append(t_fastsimplex_fit)
        results['delaunay_fit'].append(t_delaunay_fit)
        results['fastsimplex_predict'].append(t_fastsimplex_pred)
        results['delaunay_predict'].append(t_delaunay_pred)
        
        print(f"{t_fastsimplex_fit:<15.0f} {t_delaunay_fit:<15.0f} "
              f"{t_fastsimplex_pred:<15.2f} {t_delaunay_pred:<15.2f}")
    
    return results


def final_summary(construction_results, inference_results, 
                  precision_results, scalability_results):
    """
    Generate final benchmark summary
    """
    print("\n" + "="*80)
    print("FINAL SUMMARY - CAN Fast Simplex 2D COMPETE WITH DELAUNAY?")
    print("="*80)
    
    # Construction
    avg_construction_speedup = np.mean([
        construction_results['delaunay_times'][i] / construction_results['fastsimplex_times'][i]
        for i in range(len(construction_results['sizes']))
    ])
    
    print(f"\n1. CONSTRUCTION (FIT):")
    if avg_construction_speedup > 1:
        print(f"   ✅ Fast Simplex WINS: {avg_construction_speedup:.2f}x faster on average")
    else:
        print(f"   ❌ DELAUNAY WINS: {1/avg_construction_speedup:.2f}x faster on average")
    
    # Inference
    print(f"\n2. INFERENCE (PREDICT):")
    if inference_results['speedup'] > 1:
        print(f"   ✅ Fast Simplex WINS: {inference_results['speedup']:.2f}x faster")
        print(f"   Throughput: {inference_results['fastsimplex_throughput']:.0f} pred/s vs "
              f"{inference_results['delaunay_throughput']:.0f} pred/s")
    else:
        print(f"   ❌ DELAUNAY WINS: {1/inference_results['speedup']:.2f}x faster")
    
    # Precision
    print(f"\n3. PRECISION:")
    if precision_results['rmse_fastsimplex'] < precision_results['rmse_delaunay']:
        print(f"   ✅ Fast Simplex WINS: RMSE {precision_results['rmse_fastsimplex']:.6f} vs "
              f"{precision_results['rmse_delaunay']:.6f}")
    elif abs(precision_results['rmse_fastsimplex'] - precision_results['rmse_delaunay']) < 1e-6:
        print(f"   ✅ TIE: Both have RMSE ≈ {precision_results['rmse_fastsimplex']:.6f}")
    else:
        print(f"   ❌ DELAUNAY WINS: RMSE {precision_results['rmse_delaunay']:.6f} vs "
              f"{precision_results['rmse_fastsimplex']:.6f}")
    
    # Scalability
    print(f"\n4. SCALABILITY (100K points):")
    idx_100k = scalability_results['sizes'].index(100000)
    speedup_fit_100k = (scalability_results['delaunay_fit'][idx_100k] / 
                        scalability_results['fastsimplex_fit'][idx_100k])
    speedup_pred_100k = (scalability_results['delaunay_predict'][idx_100k] / 
                         scalability_results['fastsimplex_predict'][idx_100k])
    
    print(f"   Construction: {'FASTSIMPLEX' if speedup_fit_100k > 1 else 'DELAUNAY'} "
          f"{max(speedup_fit_100k, 1/speedup_fit_100k):.2f}x faster")
    print(f"   Inference:   {'FASTSIMPLEX' if speedup_pred_100k > 1 else 'DELAUNAY'} "
          f"{max(speedup_pred_100k, 1/speedup_pred_100k):.2f}x faster")
    
    # Final verdict
    print(f"\n" + "="*80)
    print("FINAL VERDICT:")
    print("="*80)
    
    construction_wins = avg_construction_speedup > 1
    inference_wins = inference_results['speedup'] > 1
    precision_wins = (precision_results['rmse_fastsimplex'] <= 
                     precision_results['rmse_delaunay'] * 1.01)  # 1% Tolerance
    
    fastsimplex_victories = sum([construction_wins, inference_wins, precision_wins])
    
    if fastsimplex_victories >= 2:
        print("✅ YES, Fast Simplex 2D CAN COMPETE WITH DELAUNAY")
        print("\nReasons:")
        if construction_wins:
            print(f"  - Faster construction ({avg_construction_speedup:.2f}x)")
        if inference_wins:
            print(f"  - Faster inference ({inference_results['speedup']:.2f}x)")
        if precision_wins:
            print(f"  - Comparable precision (RMSE ≈ {precision_results['rmse_fastsimplex']:.6f})")
    else:
        print("⚠️ Fast Simplex 2D IS COMPETITIVE BUT NOT SUPERIOR TO DELAUNAY")
        print("\nDelaunay maintains advantage in:")
        if not construction_wins:
            print(f"  - Construction ({1/avg_construction_speedup:.2f}x faster)")
        if not inference_wins:
            print(f"  - Inference ({1/inference_results['speedup']:.2f}x faster)")
        if not precision_wins:
            print(f"  - Precision (better RMSE)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("="*80)
    print("DEFINITIVE BENCHMARK: FAST SIMPLEX 2D vs SCIPY DELAUNAY")
    print("="*80)
    print("\nThis benchmark will determine if Fast Simplex 2D can compete with Delaunay")
    print("in speed, precision, and scalability.")
    
    # Silence Fast Simplex engine prints during benchmarks
    import sys
    import io
    
    # 1. Construction benchmark
    construction_res = benchmark_construction([100, 500, 1000, 5000, 10000])
    
    # 2. Inference benchmark
    inference_res = benchmark_inference_single(N=10000, n_queries=1000)
    
    # 3. Precision benchmark
    precision_res = benchmark_precision(N=5000, n_queries=500)
    
    # 4. Scalability benchmark
    scalability_res = benchmark_scalability()
    
    # Final summary
    final_summary(construction_res, inference_res, precision_res, scalability_res)
   
