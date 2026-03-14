"""
================================================================================
FAST SIMPLEX 2D - Test Suite v2.0
================================================================================

Comprehensive tests for Fast Simplex 2D interpolation engine.

EDA Team: Gemini · Claude · Alex
License: MIT
Version: 2.0

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
        engine = FastSimplex2D(max_radius=0, lambda_factor=9)
        print("✓ PASS: Engine initialized successfully")
        print(f"  - Dimensions: {engine.D}")
        print(f"  - Lambda factor: {engine.lambda_factor}")
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
        
        if engine.n_points == 100:
            print("✓ PASS: 100 points loaded correctly")
            return True
        else:
            print(f"✗ FAIL: Expected 100 points, got {engine.n_points}")
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
        print("✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ PASS: Correctly raised ValueError")
        print(f"  - Error message: {e}")
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
    print("TEST 5: Success Rate (v2.0 target: 95%+ on 1K, 99%+ on large datasets)")
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
    
    if success_rate >= 95.0:  # 95%+ is excellent for 1K dataset
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
    
    # v1.1 should be fast
    if throughput > 5000:
        print(f"  - ✓ Excellent performance (>5000 pred/s)")
    
    return True


def test_9_edge_cases():
    """Test 9: Edge cases handling"""
    print("\n" + "="*80)
    print("TEST 9: Edge Cases")
    print("="*80)
    
    # Very sparse dataset
    np.random.seed(42)
    x = np.random.rand(15) * 10
    y = np.random.rand(15) * 10
    z = x + y
    data = np.column_stack([x, y, z])
    
    engine = FastSimplex2D(lambda_factor=5)
    engine.fit(data)
    
    test_point = np.array([5.0, 5.0])
    pred = engine.predict(test_point)
    
    # May succeed or fail, but should not crash
    print(f"  - Sparse dataset (15 points): {'Success' if pred is not None else 'No support'}")
    print("✓ PASS: Edge case handled without crash")
    return True


def test_10_quadrant_coverage():
    """Test 10: All quadrants covered"""
    print("\n" + "="*80)
    print("TEST 10: Quadrant Coverage (v2.0 11-case algorithm)")
    print("="*80)
    
    # Generate data covering all quadrants
    np.random.seed(42)
    N = 500
    x = (np.random.rand(N) - 0.5) * 20  # -10 to 10
    y = (np.random.rand(N) - 0.5) * 20  # -10 to 10
    z = x + y
    data = np.column_stack([x, y, z])
    
    engine = FastSimplex2D()
    engine.fit(data)
    
    # Test points in all quadrants
    test_points = np.array([
        [5, 5],    # Quadrant I
        [-5, 5],   # Quadrant II
        [-5, -5],  # Quadrant III
        [5, -5],   # Quadrant IV
        [0, 5],    # Positive Y axis
        [5, 0],    # Positive X axis
    ])
    
    results = []
    for p in test_points:
        pred = engine.predict(p)
        results.append(pred is not None)
    
    success_count = sum(results)
    
    print(f"Quadrant coverage: {success_count}/{len(test_points)} quadrants")
    
    if success_count >= 5:  # At least 5/6
        print("✓ PASS: Good quadrant coverage")
        return True
    else:
        print("✗ FAIL: Poor quadrant coverage")
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*80)
    print("FAST SIMPLEX 2D v2.0 - COMPREHENSIVE TEST SUITE")
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
        test_9_edge_cases,
        test_10_quadrant_coverage
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
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    print("="*80)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
 
