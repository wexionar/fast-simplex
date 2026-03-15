
import numpy as np
from scipy.spatial import cKDTree

class FastSimplex2D4:
    def __init__(self, k_neighbors=15):
        self.k = k_neighbors
        self.tree = None
        self.points = None
        self.values = None

    def fit(self, data):
        # Aseguramos contigüidad para máxima velocidad de lectura
        self.points = np.ascontiguousarray(data[:, :2], dtype=np.float64)
        self.values = np.ascontiguousarray(data[:, 2], dtype=np.float64)
        self.tree = cKDTree(self.points)
        return self

    def predict(self, query_point):
        # 1. Búsqueda ultra rápida
        dist, idx = self.tree.query(query_point, k=self.k)

        # Caso exacto
        if dist[0] < 1e-12:
            return float(self.values[idx[0]])

        # 2. Centralización local
        pts = self.points[idx] - query_point
        px = pts[:, 0]
        py = pts[:, 1]
        v = self.values[idx]

        EPS = 1e-14
        n = self.k

        # 3. TRIPLE BUCLE SEMI-VECTORIZADO
        # Optimizamos el bucle interno para que NumPy maneje la lógica pesada
        for i in range(n):
            xi, yi, vi = px[i], py[i], v[i]
            for j in range(i + 1, n):
                xj, yj, vj = px[j], py[j], v[j]

                # Producto cruzado entre i y j
                cp_ij = xi * yj - yi * xj

                # Si cp_ij es casi 0, los puntos i, j y query están alineados
                if abs(cp_ij) < 1e-18: continue

                # Determinamos el signo necesario para el tercer punto
                # Para encerrar el origen, cp_jk y cp_ki deben tener el mismo signo que cp_ij
                is_pos = cp_ij > 0

                for k in range(j + 1, n):
                    xk, yk = px[k], py[k]

                    # Calculamos los otros dos productos
                    cp_jk = xj * yk - yj * xk
                    cp_ki = xk * yi - yk * xi

                    # Verificación del "Escudo" con lógica booleana rápida
                    if is_pos:
                        if cp_jk > -EPS and cp_ki > -EPS:
                            det = cp_ij + cp_jk + cp_ki
                            return float((cp_jk * vi + cp_ki * vj + cp_ij * v[k]) / det)
                    else:
                        if cp_jk < EPS and cp_ki < EPS:
                            det = cp_ij + cp_jk + cp_ki
                            return float((cp_jk * vi + cp_ki * vj + cp_ij * v[k]) / det)

        return None
       
