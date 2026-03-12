"""
================================================================================
FAST SIMPLEX 2D - Local Coordinate System for Simplex Inference
================================================================================

EDA Team: Gemini · Claude · Alex
License: MIT
Version: 1.0
Trad.: GLM

================================================================================
ALGORITHM:
1. Neighbor search (with radius R or fixed quantity)
2. Transformation to local system (Pc at origin, V1 on negative X axis)
3. Selection of V2 and V3 (x > 0, opposite signs in Y)
4. Enclosure verification
5. Prediction with barycentric coordinates

PARAMETERS:
- R: Maximum search radius (0 = no radius restriction)
- lambda_factor: Coefficient for quantity of neighbors (D * lambda_factor)
- max_neighbors: Upper limit of neighbors (safety)
================================================================================
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Dict, Union


class FastSimplex2D:
    """
    2D inference engine using local coordinate system.
    
    Features:
    - Efficient search with cKDTree
    - Normalized local coordinate system
    - Geometric simplex selection
    - Enclosure validation
    - Detailed diagnostics
    """
    
    def __init__(self, 
                 max_radius: float = 0.0,
                 lambda_factor: int = 9,
                 max_neighbors: int = 1000):
        """
        Initialize the Fast Simplex 2D engine.
        
        Args:
            max_radius: Maximum search radius. If R=0, uses fixed quantity.
            lambda_factor: Factor to calculate k neighbors = D * lambda_factor
            max_neighbors: Upper limit of neighbors (avoids extreme cases)
        """
        self.D = 2  # Fixed dimension for this version
        self.max_radius = max_radius
        self.lambda_factor = lambda_factor
        self.max_neighbors = max_neighbors
        
        # Dataset
        self.tree = None
        self.dataset_x = None
        self.dataset_y = None
        self.n_points = 0
        
    def fit(self, data: np.ndarray) -> 'FastSimplex2D':
        """
        Load dataset and build search structure.
        
        Args:
            data: Array (N, 3) where columns are [X, Y, Z]
                  X, Y: coordinates
                  Z: dependent value
        
        Returns:
            self (for chaining)
        """
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"Data must be (N, 3), received {data.shape}")
        
        self.dataset_x = data[:, :2].copy()  # Coordinates X, Y
        self.dataset_y = data[:, 2].copy()   # Values Z
        self.n_points = len(data)
        
        # Build KDTree for efficient search
        self.tree = cKDTree(self.dataset_x)
        
        print(f"✓ Fast Simplex 2D: {self.n_points} points loaded")
        print(f"  Maximum radius: {self.max_radius if self.max_radius > 0 else 'No restriction'}")
        print(f"  Lambda factor: {self.lambda_factor} (k = {self.D * self.lambda_factor})")
        print(f"  Max neighbors: {self.max_neighbors}")
        
        return self
    
    def predict(self, 
                point: np.ndarray,
                return_diagnostics: bool = False
               ) -> Union[float, Tuple[Optional[float], Dict]]:
        """
        Predict value at a point using simplex in local system.
        
        Args:
            point: Query point (x, y)
            return_diagnostics: If True, returns (prediction, diagnostics)
        
        Returns:
            If return_diagnostics=False: prediction (or None if failed)
            If return_diagnostics=True: (prediction, diagnostics_dict)
        """
        if self.tree is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        point = np.asarray(point, dtype=np.float64).flatten()
        
        if point.shape[0] != self.D:
            raise ValueError(f"Point must have {self.D} dimensions, received {len(point)}")
        
        # Initialize diagnostics
        diagnostics = {
            'success': False,
            'message': '',
            'n_neighbors_found': 0,
            'simplex_indices': None,
            'simplex_coords': None,
            'barycentric_weights': None,
            'v1_distance': None,
            'search_method': None
        }
        
        # PHASE 1: Neighbor search
        result = self._find_neighbors(point)
        
        if result is None:
            diagnostics['message'] = "Insufficient geometric support to create simplex"
            return (None, diagnostics) if return_diagnostics else None
        
        neighbor_indices, search_method = result
        diagnostics['n_neighbors_found'] = len(neighbor_indices)
        diagnostics['search_method'] = search_method
        
        # PHASE 2: Transform to local system
        V1_idx = neighbor_indices[0]
        V1_original = self.dataset_x[V1_idx]
        v1_distance = np.linalg.norm(V1_original - point)
        diagnostics['v1_distance'] = v1_distance
        
        neighbors_local = self._transform_to_local_system(
            self.dataset_x[neighbor_indices],
            origin=point,
            v1_original=V1_original
        )
        
        # PHASE 3 and 4: Select V2 and V3
        simplex_result = self._select_simplex_vertices(neighbors_local, neighbor_indices)
        
        if simplex_result is None:
            diagnostics['message'] = "Insufficient geometric support to create simplex"
            return (None, diagnostics) if return_diagnostics else None
        
        simplex_indices, V2_local, V3_local = simplex_result
        
        # PHASE 5: Verify enclosure
        simplex_coords = self.dataset_x[simplex_indices]
        simplex_values = self.dataset_y[simplex_indices]
        
        weights = self._compute_barycentric_coordinates(point, simplex_coords)
        
        if weights is None or not np.all(weights >= -1e-10):
            diagnostics['message'] = "Insufficient geometric support to create simplex"
            diagnostics['barycentric_weights'] = weights
            return (None, diagnostics) if return_diagnostics else None
        
        # PHASE 6: Prediction
        prediction = float(np.dot(weights, simplex_values))
        
        # Update success diagnostics
        diagnostics['success'] = True
        diagnostics['message'] = "Success: Simplex created and point enclosed"
        diagnostics['simplex_indices'] = simplex_indices
        diagnostics['simplex_coords'] = simplex_coords
        diagnostics['barycentric_weights'] = weights
        
        if return_diagnostics:
            return prediction, diagnostics
        
        return prediction
    
    def _find_neighbors(self, point: np.ndarray) -> Optional[Tuple[np.ndarray, str]]:
        """
        Search neighbors according to configuration (radius or fixed quantity).
        
        Returns:
            (neighbor_indices, search_method) or None if insufficient
        """
        if self.max_radius > 0:
            # Radius search
            neighbor_indices = self.tree.query_ball_point(point, r=self.max_radius)
            
            if len(neighbor_indices) < self.D + 1:
                return None
            
            # Limit to max_neighbors (safety)
            if len(neighbor_indices) > self.max_neighbors:
                # Sort by distance and take the closest ones
                distances = [np.linalg.norm(self.dataset_x[i] - point) for i in neighbor_indices]
                sorted_indices = [i for _, i in sorted(zip(distances, neighbor_indices))]
                neighbor_indices = sorted_indices[:self.max_neighbors]
            else:
                # Sort by distance
                distances = [np.linalg.norm(self.dataset_x[i] - point) for i in neighbor_indices]
                neighbor_indices = [i for _, i in sorted(zip(distances, neighbor_indices))]
            
            search_method = f"radius_search(R={self.max_radius}, found={len(neighbor_indices)})"
            
        else:
            # Fixed quantity search
            k = self.D * self.lambda_factor
            
            if self.n_points < k:
                k = self.n_points
            
            distances, neighbor_indices = self.tree.query(point, k=k)
            
            if len(neighbor_indices) < self.D + 1:
                return None
            
            search_method = f"knn_search(k={k})"
        
        return np.array(neighbor_indices), search_method
    
    def _transform_to_local_system(self, 
                                    points: np.ndarray,
                                    origin: np.ndarray,
                                    v1_original: np.ndarray) -> np.ndarray:
        """
        Transform points to the local system where:
        - origin (Pc) is at (0, 0)
        - v1 is at (-1, 0)
        
        Args:
            points: Points to transform (N, 2)
            origin: New origin (query point)
            v1_original: Original position of V1
        
        Returns:
            Transformed points (N, 2)
        """
        # 1. Translate so origin is (0, 0)
        points_centered = points - origin
        v1_centered = v1_original - origin
        
        # 2. Calculate angle to rotate V1 to (-1, 0)
        angle_v1 = np.arctan2(v1_centered[1], v1_centered[0])
        angle_target = np.pi  # 180 degrees (negative X axis)
        angle_rotation = angle_target - angle_v1
        
        # 3. Rotation matrix
        c = np.cos(angle_rotation)
        s = np.sin(angle_rotation)
        rotation_matrix = np.array([[c, -s], [s, c]])
        
        # 4. Rotate all points
        points_rotated = points_centered @ rotation_matrix.T
        
        # 5. Scale so V1 is at distance 1
        dist_v1 = np.linalg.norm(v1_centered)
        
        if dist_v1 < 1e-12:
            # V1 coincides with origin (extreme case)
            return points_rotated
        
        points_transformed = points_rotated / dist_v1
        
        return points_transformed
    
    def _select_simplex_vertices(self,
                                  neighbors_local: np.ndarray,
                                  neighbor_indices: np.ndarray
                                 ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Select V2 and V3 according to geometric criterion.
        
        Criterion:
        - V1: is already at index 0 (closest)
        - V2: x > 0, closest to the origin
        - V3: x > 0, opposite sign(y) to V2, closest
        
        Returns:
            (simplex_indices, V2_local, V3_local) or None if failed
        """
        # V1 is always the first
        V1_idx_global = neighbor_indices[0]
        
        # Search for V2: x > 0, the closest
        V2_idx_local = None
        V2_y = None
        min_dist_v2 = np.inf
        
        for i in range(1, len(neighbors_local)):
            x, y = neighbors_local[i]
            
            if x > 0:  # Criterion: right side of Pc
                dist = np.linalg.norm(neighbors_local[i])
                if dist < min_dist_v2:
                    V2_idx_local = i
                    V2_y = y
                    min_dist_v2 = dist
        
        if V2_idx_local is None:
            return None  # No neighbors with x > 0
        
        # Search for V3: x > 0, opposite sign(y) to V2, the closest
        V3_idx_local = None
        min_dist_v3 = np.inf
        
        for i in range(1, len(neighbors_local)):
            if i == V2_idx_local:
                continue
            
            x, y = neighbors_local[i]
            
            if x > 0 and np.sign(y) != np.sign(V2_y):
                dist = np.linalg.norm(neighbors_local[i])
                if dist < min_dist_v3:
                    V3_idx_local = i
                    min_dist_v3 = dist
        
        if V3_idx_local is None:
            return None  # No neighbor with opposite sign
        
        # Build simplex
        simplex_indices = np.array([
            V1_idx_global,
            neighbor_indices[V2_idx_local],
            neighbor_indices[V3_idx_local]
        ])
        
        V2_local = neighbors_local[V2_idx_local]
        V3_local = neighbors_local[V3_idx_local]
        
        return simplex_indices, V2_local, V3_local
    
    def _compute_barycentric_coordinates(self,
                                         point: np.ndarray,
                                         simplex_coords: np.ndarray
                                        ) -> Optional[np.ndarray]:
        """
        Calculate barycentric coordinates.
        
        Args:
            point: Query point (2,)
            simplex_coords: Simplex coordinates (3, 2)
        
        Returns:
            Barycentric weights [w1, w2, w3] or None if singular
        """
        V1, V2, V3 = simplex_coords
        
        # Solve: [V2-V1, V3-V1] * [w2, w3]^T = point - V1
        T = np.column_stack([V2 - V1, V3 - V1])
        
        try:
            w = np.linalg.solve(T, point - V1)
            w1 = 1.0 - w[0] - w[1]
            weights = np.array([w1, w[0], w[1]])
            return weights
        except np.linalg.LinAlgError:
            # Singular matrix (degenerate simplex)
            return None


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calcular_area_triangulo(vertices: np.ndarray) -> float:
    """
    Calculate triangle area using Shoelace formula.
    
    Args:
        vertices: Array (3, 2) with vertex coordinates
    
    Returns:
        Area of the triangle
    """
    V1, V2, V3 = vertices
    area = 0.5 * abs((V2[0] - V1[0]) * (V3[1] - V1[1]) - 
                     (V3[0] - V1[0]) * (V2[1] - V1[1]))
    return area


# ==========================================
# TESTS
# ==========================================

if __name__ == "__main__":
    print("="*80)
    print("FAST SIMPLEX 2D - Tests")
    print("="*80)
    
    # Test 1: Dense uniform dataset
    print("\n--- TEST 1: Dense uniform dataset ---")
    np.random.seed(42)
    N = 1000
    x = np.random.rand(N) * 10
    y = np.random.rand(N) * 10
    z = x + y  # Simple linear function: z = x + y
    
    data = np.column_stack([x, y, z])
    
    # Configuration 1: No radius restriction
    engine = FastSimplex2D(max_radius=0, lambda_factor=9)
    engine.fit(data)
    
    # Interior test point
    p_test = np.array([5.0, 5.0])
    pred, diag = engine.predict(p_test, return_diagnostics=True)
    
    print(f"\nQuery point: {p_test}")
    print(f"Prediction: {pred:.4f}")
    print(f"Real value: {p_test[0] + p_test[1]:.4f}")
    print(f"Error: {abs(pred - (p_test[0] + p_test[1])):.4f}")
    print(f"Diagnostic: {diag['message']}")
    print(f"Search method: {diag['search_method']}")
    print(f"Neighbors found: {diag['n_neighbors_found']}")
    print(f"V1 distance: {diag['v1_distance']:.4f}")
    print(f"Barycentric weights: {diag['barycentric_weights']}")
    
    # Test 2: With radius restriction
    print("\n--- TEST 2: With radius restriction R=2.0 ---")
    engine_r = FastSimplex2D(max_radius=2.0, lambda_factor=9, max_neighbors=50)
    engine_r.fit(data)
    
    pred_r, diag_r = engine_r.predict(p_test, return_diagnostics=True)
    
    print(f"Prediction: {pred_r:.4f}")
    print(f"Diagnostic: {diag_r['message']}")
    print(f"Search method: {diag_r['search_method']}")
    print(f"Neighbors found: {diag_r['n_neighbors_found']}")
    
    # Test 3: Sparse dataset
    print("\n--- TEST 3: Sparse dataset ---")
    N_sparse = 20
    x_sparse = np.random.rand(N_sparse) * 10
    y_sparse = np.random.rand(N_sparse) * 10
    z_sparse = x_sparse + y_sparse
    
    data_sparse = np.column_stack([x_sparse, y_sparse, z_sparse])
    
    engine_sparse = FastSimplex2D(max_radius=0, lambda_factor=5)
    engine_sparse.fit(data_sparse)
    
    pred_sparse, diag_sparse = engine_sparse.predict(p_test, return_diagnostics=True)
    
    print(f"Prediction: {pred_sparse}")
    print(f"Diagnostic: {diag_sparse['message']}")
    print(f"Success: {diag_sparse['success']}")
    
    # Test 4: Multiple points
    print("\n--- TEST 4: Batch of points ---")
    test_points = np.array([
        [2.5, 2.5],
        [5.0, 5.0],
        [7.5, 7.5],
        [1.0, 9.0]
    ])
    
    print(f"\n{'Point':<15} {'Prediction':<12} {'Real':<12} {'Error':<12} {'Status'}")
    print("-" * 70)
    
    for p in test_points:
        pred, diag = engine.predict(p, return_diagnostics=True)
        real = p[0] + p[1]
        
        if pred is not None:
            error = abs(pred - real)
            status = "✓"
        else:
            error = None
            status = "✗"
        
        print(f"{str(p):<15} {pred if pred else 'None':<12} {real:<12.4f} "
              f"{error if error else 'N/A':<12} {status}")
    
    print("\n" + "="*80)
    print("✓ Tests completed")
    print("="*80)
     
