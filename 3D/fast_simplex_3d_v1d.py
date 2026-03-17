"""
================================================================================
FAST SIMPLEX 3D - Inference Engine
================================================================================

EDA Team: Gemini · Claude · Alex
License: MIT
Version: 1.0c

================================================================================
"""

import numpy as np
import time
from scipy.spatial import cKDTree

# ==============================================================================
# MOTOR: FAST SIMPLEX 3D - VERSION 2.0 (DEFA)
# ==============================================================================
class FastSimplex3D:
    def __init__(self, neighbor_count=22):
        self.k = neighbor_count
        self.tree = None
        self.points = None
        self.values = None

    def fit(self, data):
        # data: [X, Y, Z, Value]
        self.points = np.ascontiguousarray(data[:, :3], dtype=np.float64)
        self.values = np.ascontiguousarray(data[:, 3], dtype=np.float64)
        self.tree = cKDTree(self.points)
        return self

    def predict(self, query_point):
        dist, idx = self.tree.query(query_point, k=self.k)
        
        if dist[0] < 1e-12:
            return float(self.values[idx[0]])

        # 1. Centrado local respecto al query point
        pts = self.points[idx] - query_point
        v = self.values[idx]
        
        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
        n = self.k
        EPS = 1e-15

        # 2. Bucle Triple con Salida Vectorizada para el 4to punto
        for i in range(n):
            xi, yi, zi, vi = px[i], py[i], pz[i], v[i]
            for j in range(i + 1, n):
                xj, yj, zj, vj = px[j], py[j], pz[j], v[j]
                
                # Pre-calculamos productos cruzados comunes (i x j)
                cx_ij = yi * zj - zi * yj
                cy_ij = zi * xj - xi * zj
                cz_ij = xi * yj - yi * xj
                
                for k in range(j + 1, n):
                    xk, yk, zk, vk = px[k], py[k], pz[k], v[k]
                    
                    # Determinante (Volumen orientado) de (i, j, k)
                    cp_ijk = xk * cx_ij + yk * cy_ij + zk * cz_ij
                    
                    if abs(cp_ijk) < 1e-18: continue
                    is_pos = cp_ijk > 0
                    
                    # --- VECTORIZACIÓN DEL CUARTO PUNTO (l) ---
                    l_idx = np.arange(k + 1, n)
                    xl, yl, zl = px[l_idx], py[l_idx], pz[l_idx]
                    
                    # Determinantes de las caras opuestas
                    cp_ijl = xl * cx_ij + yl * cy_ij + zl * cz_ij
                    
                    cx_ik, cy_ik, cz_ik = (yi * zk - zi * yk), (zi * xk - xi * zk), (xi * yk - yi * xk)
                    cp_ikl = xl * cx_ik + yl * cy_ik + zl * cz_ik
                    
                    cx_jk, cy_jk, cz_jk = (yj * zk - zj * yk), (zj * xk - xj * zk), (xj * yk - yj * xk)
                    cp_jkl = xl * cx_jk + yl * cy_jk + zl * cz_jk

                    # Verificación de signos para encapsulamiento (Q dentro de IJKL)
                    if is_pos:
                        valid = (cp_jkl > -EPS) & (cp_ikl < EPS) & (cp_ijl > -EPS)
                    else:
                        valid = (cp_jkl < EPS) & (cp_ikl > -EPS) & (cp_ijl < EPS)
                    
                    found = np.where(valid)[0]
                    if len(found) > 0:
                        idx_l = found[0]
                        w_i, w_j, w_k, w_l = abs(cp_jkl[idx_l]), abs(cp_ikl[idx_l]), abs(cp_ijl[idx_l]), abs(cp_ijk)
                        return float((w_i*vi + w_j*vj + w_k*vk + w_l*v[l_idx[idx_l]]) / (w_i + w_j + w_k + w_l))
                            
        return None

# ==============================================================================
# SCRIPT DE EVALUACIÓN (TEST EDA TEAM)
# ==============================================================================

def test_function(x, y, z):
    # Función curva compleja: senos, cosenos e interacciones
    return np.sin(3*x) + np.cos(3*y) + 0.5*z + x*y - y*z

print("--- FAST SIMPLEX 3D v2.0 - INICIANDO TEST ---")
N_POINTS = 100000
N_QUERIES = 2000 # Subimos las consultas para estresar el motor

# Generación de Datos
np.random.seed(42)
xyz = np.random.rand(N_POINTS, 3)
values = test_function(xyz[:,0], xyz[:,1], xyz[:,2])
data = np.column_stack([xyz, values])

# Construcción
print(f"Puntos: {N_POINTS} | Construyendo motor...")
t_start_build = time.time()
engine = FastSimplex3D(neighbor_count=22)
engine.fit(data)
t_build = time.time() - t_start_build

# Consultas
print(f"Consultas: {N_QUERIES} | Ejecutando...")
queries = np.random.rand(N_QUERIES, 3)
true_values = test_function(queries[:,0], queries[:,1], queries[:,2])

t_start_query = time.time()
results = [engine.predict(q) for q in queries]
t_query = time.time() - t_start_query

# Procesamiento de Resultados
results = np.array(results)
mask = np.array([r is not None for r in results])
valid_res = results[mask].astype(float)
valid_count = np.sum(mask)

if valid_count > 0:
    errors = np.abs(valid_res - true_values[mask])
    mean_err = np.mean(errors)
    max_err = np.max(errors)
else:
    mean_err = max_err = 0

# --- PRINT FINAL ---
print("\n" + "="*40)
print("      RESULTADOS FAST SIMPLEX 3D v2.0")
print("="*40)
print(f"Tiempo Construcción: {t_build:.4f} s")
print(f"Tiempo Consultas:    {t_query:.4f} s")
print(f"Throughput:          {N_QUERIES/t_query:.0f} pred/s")
print("-"*40)
print(f"Éxito (Cobertura):   {100 * valid_count / N_QUERIES:.2f} %")
print(f"Error Medio (Mean):  {mean_err:.8f}")
print(f"Error Máximo (Max):  {max_err:.8f}")
print("="*40)
 
