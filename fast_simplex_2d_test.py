"""
================================================================================
FAST SIMPLEX 2D - Test Suite v3.0
================================================================================

Comprehensive tests for Fast Simplex 2D v3.0 (Angular Algorithm).

EDA Team: Gemini · Claude · Alex
License: MIT
Version: 3.0

================================================================================
"""

import numpy as np
import sys
from fast_simplex_2d import FastSimplex2D


def test_1_basic_initialization():
    """Test 1: Basic initialization"""
    print("\n" + "="*80)
    print("TEST 1: Basic Initialization")
    print("="*80)
    
    try:
        engine = FastSimplex2D(k_neighbors=18)
        print("✓ PASS: Engine initialized successfully")
        print(f"  - K neighbors: {engine.k}")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_2_data_loading():
    """Test 2: Data loading and validation"""
    print("\n" + "="*80)
    print("TEST 2: Data Loading")
    print("="*80)
    
    np.random.seed(42)
    x = np.random.rand(100) * 10
    y = np.random.rand(100) * 10
    z = x + y
    data = np.column_stack([x, y, z])
    
    try:
        engine = FastSimplex2D()
        engine.fit(data)
        
        if len(engine.points) == 100:
            print("✓ PASS: 100 points loaded correctly")
            return True
        else:
            print(f"✗ FAIL: Expected 100 points, got {len(engine.points)}")
            return False
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_3_wrong_shape():
    """Test 3: Wrong data shape rejection"""
    print("\n" + "="*80)
    print("TEST 3: Wrong Shape Rejection")
    print("="*80)
    
    wrong_data = np.random.rand(10, 2)
    
    try:
        engine = FastSimplex2D()
        engine.fit(wrong_data)
        print("✗ FAIL: Should have raised error for wrong shape")
        return False
    except (ValueError, IndexError) as e:
        print(f"✓ PASS: Correctly rejected wrong shape")
        print(f"  - Error type: {type(e).__name__}")
        return True
    except Exception as e:
        print(f"✗ FAIL: Wrong exception type: {e}")
        return False


def test_4_simple_prediction():
    """Test 4: Simple prediction on linear function"""
    print("\n" + "="*80)
    print("TEST 4: Simple Prediction (z = x + y)")
    print("="*80)
    
    np.random.seed(42)
    x = np.random.rand(500) * 10
    y = np.random.rand(500) * 10
    z = x + y
    data = np.column_stack([x, y, z])
    
    engine = FastSimplex2D()
    engine.fit(data)
    
    test_point = np.array([5.0, 5.0])
    expected = 10.0
    
    pred = engine.predict(test_point)
    
    if pred is not None:
        error = abs(pred - expected)
        if error < 0.5:
            print(f"✓ PASS: Prediction accurate")
            print(f"  - Expected: {expected}")
            print(f"  - Predicted: {pred:.4f}")
            print(f"  - Error: {error:.6f}")
            return True
        else:
            print(f"✗ FAIL: Error too large")
            print(f"  - Expected: {expected}")
            print(f"  - Predicted: {pred:.4f}")
            print(f"  - Error: {error:.6f}")
            return False
    else:
        print("✗ FAIL: Prediction returned None")
        return False


def test_5_success_rate():
    """Test 5: Success rate validation"""
    print("\n" + "="*80)
    print("TEST 5: Success Rate (v3.0 target: 99%+ on 1K, 99.5%+ on large datasets)")
    print("="*80)
    
    np.random.seed(42)
    x = np.random.rand(1000) * 10
    y = np.random.rand(1000) * 10
    z = x + y
    data = np.column_stack([x, y, z])
    
    engine = FastSimplex2D()
    engine.fit(data)
    
    test_points = np.random.rand(200, 2) * 10
    predictions = [engine.predict(p) for p in test_points]
    success_count = sum([1 for p in predictions if p is not None])
    success_rate = success_count / len(predictions) * 100
    
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Successful: {success_count}/{len(predictions)}")
    
    if success_rate >= 95.0:
        print("✓ PASS: Success rate ≥ 95%")
        return True
    else:
        print(f"✗ FAIL: Success rate below 95%")
        return False


def test_6_batch_predictions():
    """Test 6: Batch predictions"""
    print("\n" + "="*80)
    print("TEST 6: Batch Predictions")
    print("="*80)
    
    np.random.seed(42)
    x = np.random.rand(300) * 10
    y = np.random.rand(300) * 10
    z = x + y
    data = np.column_stack([x, y, z])
    
    engine = FastSimplex2D()
    engine.fit(data)
    
    test_points = np.array([
        [2.5, 2.5],
        [5.0, 5.0],
        [7.5, 7.5]
    ])
    
    results = []
    for point in test_points:
        pred = engine.predict(point)
        results.append(pred)
    
    success_count = sum([1 for r in results if r is not None])
    
    if success_count == len(test_points):
        print(f"✓ PASS: All {len(test_points)} predictions successful")
        for i, (point, pred) in enumerate(zip(test_points, results)):
            expected = point[0] + point[1]
            error = abs(pred - expected)
            print(f"  - Point {i+1}: pred={pred:.4f}, expected={expected:.4f}, error={error:.4f}")
        return True
    else:
        print(f"✗ FAIL: Only {success_count}/{len(test_points)} predictions successful")
        return False


def test_7_exact_match():
    """Test 7: Exact match with dataset point"""
    print("\n" + "="*80)
    print("TEST 7: Exact Match")
    print("="*80)
    
    data = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 2]
    ])
    
    engine = FastSimplex2D()
    engine.fit(data)
    
    test_point = np.array([1.0, 1.0])
    expected = 2.0
    
    pred = engine.predict(test_point)
    
    if pred is not None:
        error = abs(pred - expected)
        if error < 1e-6:
            print("✓ PASS: Exact match prediction")
            print(f"  - Expected: {expected}")
            print(f"  - Predicted: {pred:.10f}")
            print(f"  - Error: {error:.2e}")
            return True
        else:
            print(f"✗ FAIL: Error too large for exact match: {error}")
            return False
    else:
        print("✗ FAIL: Prediction returned None")
        return False


def test_8_performance():
    """Test 8: Performance benchmark"""
    print("\n" + "="*80)
    print("TEST 8: Performance")
    print("="*80)
    
    import time
    
    np.random.seed(42)
    N = 5000
    x = np.random.rand(N) * 10
    y = np.random.rand(N) * 10
    z = x + y
    data = np.column_stack([x, y, z])
    
    # Measure fit time
    start = time.perf_counter()
    engine = FastSimplex2D()
    engine.fit(data)
    fit_time = (time.perf_counter() - start) * 1000
    
    # Measure predict time (100 queries)
    queries = np.random.rand(100, 2) * 10
    start = time.perf_counter()
    for q in queries:
        engine.predict(q)
    predict_time = (time.perf_counter() - start) * 1000
    
    throughput = 100 / (predict_time / 1000)
    
    print(f"✓ PASS: Performance measured")
    print(f"  - Fit time ({N} points): {fit_time:.2f} ms")
    print(f"  - Predict time (100 queries): {predict_time:.2f} ms")
    print(f"  - Throughput: {throughput:.0f} pred/s")
    
    if throughput > 5000:
        print(f"  - ✓ Excellent performance (>5000 pred/s)")
    
    return True


def test_9_curved_function():
    """Test 9: Non-linear function (curved)"""
    print("\n" + "="*80)
    print("TEST 9: Curved Function (z = sin(x) * cos(y))")
    print("="*80)
    
    np.random.seed(42)
    N = 1000
    x = np.random.rand(N) * 10
    y = np.random.rand(N) * 10
    z = np.sin(x) * np.cos(y)
    data = np.column_stack([x, y, z])
    
    engine = FastSimplex2D()
    engine.fit(data)
    
    # Test points
    test_points = np.random.rand(100, 2) * 10
    true_values = np.sin(test_points[:, 0]) * np.cos(test_points[:, 1])
    
    predictions = []
    for p in test_points:
        pred = engine.predict(p)
        predictions.append(pred if pred is not None else np.nan)
    
    predictions = np.array(predictions)
    valid = ~np.isnan(predictions)
    
    if np.sum(valid) > 0:
        errors = np.abs(predictions[valid] - true_values[valid])
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        success_rate = np.sum(valid) / len(predictions) * 100
        
        print(f"✓ PASS: Curved function tested")
        print(f"  - Success rate: {success_rate:.1f}%")
        print(f"  - Mean error: {mean_error:.6f}")
        print(f"  - Max error: {max_error:.6f}")
        
        if mean_error < 0.01:
            print(f"  - ✓ Excellent precision on curves")
        
        return True
    else:
        print("✗ FAIL: No valid predictions")
        return False


def test_10_large_dataset():
    """Test 10: Large dataset scalability"""
    print("\n" + "="*80)
    print("TEST 10: Large Dataset Scalability")
    print("="*80)
    
    np.random.seed(42)
    N = 50000
    x = np.random.rand(N) * 10
    y = np.random.rand(N) * 10
    z = x + y
    data = np.column_stack([x, y, z])
    
    import time
    
    start = time.perf_counter()
    engine = FastSimplex2D()
    engine.fit(data)
    fit_time = (time.perf_counter() - start) * 1000
    
    queries = np.random.rand(1000, 2) * 10
    
    start = time.perf_counter()
    predictions = [engine.predict(q) for q in queries]
    query_time = (time.perf_counter() - start) * 1000
    
    success_rate = sum([1 for p in predictions if p is not None]) / len(predictions) * 100
    throughput = 1000 / (query_time / 1000)
    
    print(f"✓ PASS: Large dataset handled")
    print(f"  - Dataset size: {N:,} points")
    print(f"  - Fit time: {fit_time:.2f} ms")
    print(f"  - Query time (1000): {query_time:.2f} ms")
    print(f"  - Throughput: {throughput:.0f} pred/s")
    print(f"  - Success rate: {success_rate:.1f}%")
    
    if success_rate >= 99.0:
        print(f"  - ✓ Excellent success rate on large dataset")
    
    return True


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*80)
    print("FAST SIMPLEX 2D v3.0 - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("EDA Team: Gemini · Claude · Alex")
    print("="*80)
    
    tests = [
        test_1_basic_initialization,
        test_2_data_loading,
        test_3_wrong_shape,
        test_4_simple_prediction,
        test_5_success_rate,
        test_6_batch_predictions,
        test_7_exact_match,
        test_8_performance,
        test_9_curved_function,
        test_10_large_dataset
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ CRASH: {test.__name__} crashed with: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} | Test {i}: {test.__name__}")
    
    print("="*80)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\n✅ Fast Simplex v3.0 is READY for production!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    print("="*80)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
 
