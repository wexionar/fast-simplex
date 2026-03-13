# Fast Simplex 2D

⚡ **Fast geometric interpolation using local coordinate systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

---

## 🚀 **Overview**

**Fast Simplex** is a 2D interpolation engine that constructs simplexes (triangles) using a novel **local coordinate system approach**. It achieves **7-40x faster construction** and competitive query performance compared to Scipy's Delaunay triangulation.

### **Key Features:**

- ⚡ **Blazing Fast Construction:** 7-40x faster than Delaunay
- 📈 **Scales Well:** Performance advantage grows with dataset size
- 🎯 **Geometrically Honest:** Rejects predictions when geometric support is insufficient
- 🔧 **Simple API:** Easy to use, similar to scikit-learn
- 📊 **Diagnostic Support:** Optional detailed diagnostics for every prediction

---

## 📊 **Benchmarks vs Scipy Delaunay**

### **Construction Speed (N=10,000 points):**

```
Fast Simplex:   3.55 ms
Scipy Delaunay: 60.61 ms
→ Fast Simplex is 17.1x FASTER ✅
```

### **Inference Speed (1,000 queries):**

```
Fast Simplex:   132.04 ms  (7,574 pred/s)
Scipy Delaunay: 165.52 ms  (6,042 pred/s)
→ Fast Simplex is 1.3x FASTER ✅
```

### **Scalability (100,000 points):**

| Metric | Fast Simplex | Delaunay | Speedup |
|--------|-------------|----------|---------|
| **Construction** | 86.1 ms | 599.1 ms | **7.0x** |
| **100 Queries** | 12.4 ms | 1101.1 ms | **89.0x** 🚀 |

**Full benchmarks:** See [benchmarks/](./benchmarks/) directory

---

## 🛠️ **Installation**

### **From Source:**

```bash
git clone https://github.com/wexionar/fast-simplex.git
cd fast-simplex
```

### **Requirements:**

```bash
pip install numpy scipy
```

**Python version:** 3.7+

---

## 🎯 **Quick Start**

```python
import numpy as np
from fast_simplex_2d import FastSimplex2D

# Generate sample data
x = np.random.rand(1000) * 10
y = np.random.rand(1000) * 10
z = x + y  # Function to interpolate
data = np.column_stack([x, y, z])

# Create and fit engine
engine = FastSimplex2D()
engine.fit(data)

# Predict at new point
point = np.array([5.0, 5.0])
prediction = engine.predict(point)
print(f"Prediction: {prediction}")  # ~10.0
```

---

## 📖 **Usage Examples**

### **Example 1: Basic Usage**

```python
from fast_simplex_2d import FastSimplex2D
import numpy as np

# Your data: [X, Y, Z] where Z = f(X, Y)
data = np.array([
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 2]
])

# Fit
engine = FastSimplex2D()
engine.fit(data)

# Predict
result = engine.predict([0.5, 0.5])
print(result)  # ~1.0
```

---

### **Example 2: With Diagnostics**

```python
# Get detailed diagnostic information
prediction, diagnostics = engine.predict([5.0, 5.0], return_diagnostics=True)

print(diagnostics['success'])              # True/False
print(diagnostics['message'])              # Status message
print(diagnostics['barycentric_weights'])  # Interpolation weights
print(diagnostics['v1_distance'])          # Distance to nearest neighbor
```

---

### **Example 3: Radius Restriction**

```python
# Limit search to neighbors within radius R
engine = FastSimplex2D(max_radius=2.0, max_neighbors=50)
engine.fit(data)

prediction = engine.predict([5.0, 5.0])
```

---

### **Example 4: Batch Predictions**

```python
# Predict multiple points
test_points = np.array([
    [2.5, 2.5],
    [5.0, 5.0],
    [7.5, 7.5]
])

predictions = [engine.predict(p) for p in test_points]
```

---

## ⚙️ **API Reference**

### **FastSimplex2D**

```python
FastSimplex2D(max_radius=0.0, lambda_factor=9, max_neighbors=1000)
```

**Parameters:**

- `max_radius` (float): Maximum search radius. If 0, uses fixed neighbor count.
- `lambda_factor` (int): Neighbor count multiplier. `k = D * lambda_factor` (default: 18 neighbors in 2D)
- `max_neighbors` (int): Safety limit to avoid extreme cases.

**Methods:**

#### `fit(data)`

Load dataset and build spatial index.

- **Parameters:**
  - `data` (ndarray): Shape (N, 3). Columns: [X, Y, Z]
- **Returns:** `self` (for method chaining)

#### `predict(point, return_diagnostics=False)`

Predict value at a query point.

- **Parameters:**
  - `point` (array-like): Query point [x, y]
  - `return_diagnostics` (bool): Return detailed diagnostics
- **Returns:**
  - If `return_diagnostics=False`: `float` or `None`
  - If `return_diagnostics=True`: `(float, dict)` or `(None, dict)`

**Diagnostics Dictionary:**

```python
{
    'success': bool,                    # True if prediction succeeded
    'message': str,                     # Status message
    'n_neighbors_found': int,           # Number of neighbors found
    'simplex_indices': array,           # Indices of simplex vertices
    'simplex_coords': array,            # Coordinates of simplex
    'barycentric_weights': array,       # Barycentric interpolation weights
    'v1_distance': float,               # Distance to nearest neighbor
    'search_method': str                # Search method used
}
```

---

## 🧪 **Running Tests**

```bash
python fast_simplex_2d_test.py
```

**Expected output:**

```
✓ PASS | Test 1: test_1_initialization
✓ PASS | Test 2: test_2_data_loading
✓ PASS | Test 3: test_3_wrong_shape
...
TOTAL: 10/10 tests passed
🎉 ALL TESTS PASSED! 🎉
```

---

## 🎮 **When to Use Fast Simplex**

### ✅ **Use Fast Simplex when:**

- Dataset is large (N > 10,000 points)
- You need fast construction (e.g., iterative training, cross-validation)
- You're building real-time APIs or embedded systems
- Performance matters more than guaranteed coverage

### ⚠️ **Consider Scipy Delaunay when:**

- You need 99.5%+ success rate (Fast Simplex v1.0: ~96%)
- Dataset is very small (N < 1,000)
- You require mathematical guarantee of global optimal simplex

---

## 📐 **How It Works**

Fast Simplex uses a novel **local coordinate system** approach:

1. **Find nearest neighbor (V1)** using KDTree (O(log N))
2. **Create local coordinate system:**
   - Origin: Query point Pc → (0, 0)
   - V1 → (-1, 0) on X-axis
3. **Select V2 and V3:**
   - V2: Nearest neighbor with `x > 0` (right of Pc)
   - V3: Nearest neighbor with `x > 0` AND opposite Y sign
4. **Verify encapsulation** using barycentric coordinates
5. **Interpolate** if point is inside simplex

**Result:** Geometrically coherent simplex without full triangulation.

---

## 🔬 **Algorithm Details**

### **Complexity:**

- **Construction:** O(N log N) for KDTree (faster than Delaunay's triangulation)
- **Query:** O(log N) + O(k) where k=18 (constant)

### **Precision:**

- **RMSE on linear functions:** ~0.000000 (identical to Delaunay)
- **Success rate:** ~96% (v1.0)

### **Why is it faster?**

**Delaunay:**
- Pre-computes **entire** triangulation (expensive)
- Fast queries but slow construction

**Fast Simplex:**
- Builds **only** spatial index (KDTree)
- Constructs simplexes **on-demand**
- Fast construction, competitive queries

---

## 🔧 **Current Status: Version 1.0**

This is **version 1.0** - a solid foundation with **room for improvement**. We're being honest about what works and what can be better.

### **What Works Well:**

✅ **Construction speed:** 7-40x faster than Delaunay  
✅ **Scalability:** Advantage grows with dataset size  
✅ **Precision:** Identical RMSE to Delaunay on linear functions  
✅ **Simplicity:** Clean API, easy to use  

### **Known Limitations (and our plans):**

1. **Success Rate: ~96% (vs Delaunay's ~99.5%)**
   - **Why:** Conservative geometric criteria (prioritizes nearest neighbors)
   - **v1.1 goal:** Improve simplex selection for more stable triangles
   - **v1.2 goal:** Expand valid region for simplex creation in sparse areas

2. **Simplex Quality**
   - **Current:** Selects nearest available neighbors
   - **Future:** Add quality metrics (aspect ratio, angle constraints)
   - **Impact:** More robust predictions in challenging geometries

3. **Coverage in Sparse Regions**
   - **Current:** Rejects queries when geometric criteria not met
   - **Future:** Adaptive strategies (wider search, degradation to IDW)
   - **Impact:** Higher success rate without sacrificing speed

### **Development Roadmap:**

- 🔄 **v1.1:** Improved simplex stability (Q2 2026)
- 🔄 **v1.2:** Extended coverage strategies (Q3 2026)
- 🔄 **v2.0:** 3D support (Q4 2026)
- 🔬 **Research:** nD generalization (exploratory, no timeline)

**We follow the SLRM philosophy:**
> "Honesty over promises. We ship what works, and clearly state what needs improvement."

---

## 📚 **Documentation**

- **Test Suite:** `fast_simplex_2d_test.py`
- **Benchmarks:** See `benchmarks/` directory
- **Algorithm Guide:** See `docs/` directory

---

## 👥 **Team**

**EDA Team:**
- **Gemini** - Algorithm Exploration
- **Claude** - Implementation & Benchmarks
- **Alex** - Geometric Design

Part of the [SLRM](https://github.com/wexionar/abc-slrm) (Simplex Localized Regression Models) initiative.

---

## 📜 **License**

MIT License - see [LICENSE](./LICENSE) file for details.

---

## 🙏 **Philosophy**

This project follows the SLRM philosophy:

> **"Geometric honesty over artificial precision.**  
> **If we don't have good support, we say so.**  
> **Version 1.0 means: it works, and we know how to make it better."**

We believe in:
- 🎯 **Shipping working code** over waiting for perfection
- 📊 **Transparent benchmarks** over marketing claims
- 🔧 **Clear limitations** over hidden caveats
- 🚀 **Iterative improvement** over one-shot releases

---

## 📫 **Contact & Contributing**

- **Repository:** https://github.com/wexionar/fast-simplex
- **Issues:** Bug reports and feature requests welcome
- **Contributing:** PRs welcome! Please include tests.

---

## 🌟 **Give us a Star!**

If you find Fast Simplex useful, please consider giving it a ⭐ on GitHub!

It helps others discover the project and motivates us to keep improving it.

---

**Made with geometric precision by EDA Team** 📐✨  
**v1.0 - A solid start, with more to come** 🚀
 
