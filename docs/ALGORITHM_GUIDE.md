# Fast Simplex 2D - User Guide

## 📐 **Algorithm Concept**

This algorithm builds a simplex (triangle in 2D) using a **local coordinate system** centered on the query point.

### **Philosophy:**

```
Instead of searching for "the globally optimal simplex" (costly/impossible),
we build a "geometrically coherent" simplex in local coordinates.
```

---

## 🎯 **Algorithm Steps:**

### **1. Neighbor Search**

```python
if R > 0:
    # Search for ALL neighbors within radius R
    # Limit to max_neighbors (safety, avoids extreme cases)
else:  # R == 0
    # Search fixed quantity: k = D * λ
    # Example: D=2, λ=9 → k=18 neighbors
```

**Professionalism:** The `max_neighbors` parameter prevents loading thousands of neighbors unnecessarily in ultra-dense datasets.

---

### **2. Local Coordinate System**

```
Original:                Local (transformed):
                        
   V1 ●                      
    \                           
     \                            V3 ●
      Pc ●                   y ↑  /
                                | /
                       V1 ●----Pc--+-- → x
                       -1,0   0,0 \
                                   \
                                 V2 ●

Transformation:
- Pc (query point) → (0, 0)
- V1 (nearest neighbor) → (-1, 0)
- All other neighbors rotate and scale equally
```

**Advantage:** Normalized scale (distance V1-Pc = 1), aligned X axis.

---

### **3. Selection of V2 and V3**

```python
# V2: Nearest neighbor with x > 0 (right side of Pc)
for neighbor in local_neighbors:
    if x > 0:
        V2 = neighbor
        break

# V3: Nearest neighbor with x > 0 AND opposite sign(y) to V2
for neighbor in local_neighbors:
    if x > 0 and sign(y) != sign(y_V2):
        V3 = neighbor
        break
```

**Result:** Pc is "sandwiched" between V2 (up/down) and V3 (down/up).

---

### **4. Verification and Prediction**

```python
# Barycentric coordinates
weights = calculate_barycentric(Pc, simplex)

if all(weights >= 0):
    # Pc is encapsulated ✅
    prediction = dot(weights, simplex_values)
else:
    # Failure (should not happen if algorithm correct)
    return None
```

---

## 💻 **Code Usage:**

### **Case 1: Dense dataset without radius restriction**

```python
from fast_simplex_2d import FastSimplex2D
import numpy as np

# Create dataset
N = 1000
x = np.random.rand(N) * 10
y = np.random.rand(N) * 10
z = x + y  # Example function

data = np.column_stack([x, y, z])

# Configure engine (no radius, search by quantity)
engine = FastSimplex2D(
    max_radius=0,        # No radius restriction
    lambda_factor=9,     # k = D * 9 = 18 neighbors
    max_neighbors=1000   # Safety limit
)

engine.fit(data)

# Predict
point = np.array([5.0, 5.0])
prediction = engine.predict(point)

print(f"Prediction: {prediction}")
```

---

### **Case 2: Dataset with radius restriction**

```python
# Configure engine (radius search)
engine = FastSimplex2D(
    max_radius=2.0,      # Only neighbors within radius 2.0
    lambda_factor=9,     # Not used (R > 0)
    max_neighbors=50     # Limit: maximum 50 neighbors
)

engine.fit(data)

# Predict
prediction, diagnostics = engine.predict(point, return_diagnostics=True)

print(f"Prediction: {prediction}")
print(f"Neighbors found: {diagnostics['n_neighbors_found']}")
print(f"Method: {diagnostics['search_method']}")
```

---

### **Case 3: With complete diagnostics**

```python
pred, diag = engine.predict(point, return_diagnostics=True)

if diag['success']:
    print(f"✓ Prediction: {pred:.4f}")
    print(f"  Distance to V1: {diag['v1_distance']:.4f}")
    print(f"  Barycentric weights: {diag['barycentric_weights']}")
    print(f"  Simplex indices: {diag['simplex_indices']}")
else:
    print(f"✗ Failure: {diag['message']}")
```

---

## 🔧 **Recommended Parameters:**

### **For DENSE datasets (N > 1000, uniform distribution):**

```python
FastSimplex2D(
    max_radius=0,        # No restriction
    lambda_factor=9,     # k=18 neighbors (enough)
    max_neighbors=100    # Conservative limit
)
```

---

### **For SPARSE datasets (N < 100, irregular distribution):**

```python
FastSimplex2D(
    max_radius=0,        # No restriction
    lambda_factor=15,    # k=30 neighbors (more options)
    max_neighbors=50     # Doesn't matter much (small dataset)
)
```

---

### **For datasets with known EMPTY ZONES:**

```python
FastSimplex2D(
    max_radius=5.0,      # Known maximum radius
    lambda_factor=9,     # Not used
    max_neighbors=200    # Depends on expected density
)
```

---

## ⚙️ **Professional Details Implemented:**

### **1. Safety limit (`max_neighbors`):**

```python
# If R=3.0 and there are 5000 points inside:
# → Take only the 50 closest (max_neighbors=50)
# → Avoids unnecessary processing
```

**Reason:** In ultra-dense datasets or large radii, there may be thousands of neighbors. Limiting improves performance without losing quality.

---

### **2. Handling of `lambda_factor` (λ):**

```python
# λ = neighbor multiplier
# k = D * λ

# Examples:
D=2, λ=9  → k=18 neighbors
D=2, λ=15 → k=30 neighbors (more options, slower)
D=3, λ=9  → k=27 neighbors (for future 3D)
```

**Note:** I used `lambda_factor` instead of `λ` because `lambda` is a reserved keyword in Python.

---

### **3. Sorting by distance:**

```python
# When R > 0:
# 1. Search ALL in radius R
# 2. Sort them by distance
# 3. Take the closest max_neighbors
# 4. THEN select V1, V2, V3

# Guarantees we always work with the closest ones
```

---

### **4. Detailed diagnostics:**

```python
diagnostics = {
    'success': bool,              # True if success
    'message': str,               # Descriptive message
    'n_neighbors_found': int,     # Quantity of neighbors
    'simplex_indices': array,     # Simplex indices
    'simplex_coords': array,      # Simplex coordinates
    'barycentric_weights': array, # Barycentric weights
    'v1_distance': float,         # Distance to nearest neighbor
    'search_method': str          # "knn_search(k=18)" or "radius_search(R=2.0)"
}
```

**Useful for:** Debugging, validation, quality reports.

---

## 🚀 **Test Results:**

### **Test 1: Dense uniform dataset (N=1000)**

```
Query point: [5.0, 5.0]
Real function: z = x + y = 10.0

Prediction: 10.0000
Error: 0.0000 ✓

Diagnostics:
- Neighbors: 18 (knn_search)
- V1 Distance: 0.1357
- Weights: [0.725, 0.080, 0.195]
```

**Perfect:** Zero error on linear function.

---

### **Test 2: With radius R=2.0**

```
Method: radius_search(R=2.0, found=50)
Neighbors found: 50 (limited to max_neighbors)
Prediction: 10.0000 ✓
```

**Correct:** Finds enough neighbors, limits to 50.

---

### **Test 3: Sparse dataset (N=20)**

```
k = D * λ = 2 * 5 = 10 neighbors
Dataset: 20 points

Prediction: 10.0 ✓
Success: True
```

**Robust:** Works even with few points.

---

## 📊 **Comparison with Other Methods:**

| Method | Search Complexity | Simplex | Enclosure |
|--------|---------------------|---------|-----------------|
| **Delaunay** | O(N log N) construction | Global optimum | Guaranteed (if in convex hull) |
| **Local System (this)** | O(log N) | Local geometric | Explicitly verified |
| **Axial Simplex** | O(log N) | By axes | Axial (can be asymmetric) |

**Advantage of Local System:**
- ✅ Fast like Axial Simplex (O(log N))
- ✅ More symmetric than Axial Simplex
- ✅ More flexible than Delaunay (works with R)

---

## 🔮 **Extension to 3D (Future):**

```python
# Same philosophy:
# - Pc at (0, 0, 0)
# - V1 at (-1, 0, 0)
# - V2, V3, V4 with x > 0

# Additional criteria:
# - V2: x > 0 (nearest)
# - V3: x > 0, y * y_V2 < 0 (opposite y sign)
# - V4: x > 0, z * z_V2 < 0 AND z * z_V3 < 0 (combine signs)

# Result: Tetrahedron that encapsulates Pc
```

---

## ⚠️ **Known Limitations:**

### **1. Extreme anisotropic datasets**

```
If all points are on a line:
● ● ● Pc ● ● ●

Failure: No neighbors with opposite sign_y
Message: "Insufficient geometric support"
```

**Future solution:** Downgrade to LOGOS (1D interpolation).

---

### **2. Very sparse regions**

```
If the nearest neighbor is very far:
V1_distance = 10.0 (on dataset scale [0, 1])

Result: Very large simplex (high uncertainty)
```

**Solution:** Use diagnostics to warn (`v1_distance > threshold`).

---

### **3. Points near the convex hull border**

```
If Pc is on the frontier:
- There may be no neighbors with x > 0 in some direction
- Failure: "Insufficient geometric support"
```

**This is CORRECT:** Reject instead of extrapolating.

---

## 🎯 **Conclusion:**

This algorithm is:
- ✅ **Efficient:** O(log N) search
- ✅ **Geometrically coherent:** Normalized local system
- ✅ **Robust:** Handles edge cases professionally
- ✅ **Diagnostic:** Detailed information for validation
- ✅ **Extensible:** Ready for 3D

**Ideal for:** 2D datasets with good local density, where a compact and symmetric simplex is sought.

---

**EDA Team:** Gemini · Claude · Alex<br>
**License:** MIT<br>
**Version:** 1.0<br>
**Trans.:** GLM<br>
 
