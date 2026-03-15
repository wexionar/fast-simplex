
import numpy as np
import time
from scipy.spatial import cKDTree

class FastSimplex2DAngular_Final3:
    def __init__(self, k_neighbors=15):
        self.k = k_neighbors
        self.tree = None
        self.points = None
        self.values = None

    def fit(self, data):
        # data: (N, 3) -> [x, y, valor]
        self.points = np.ascontiguousarray(data[:, :2], dtype=np.float64)
        self.values = np.ascontiguousarray(data[:, 2], dtype=np.float64)
        self.tree = cKDTree(self.points)
        return self

    def predict(self, query_point):
        # El árbol ya devuelve los índices ordenados por distancia (v1.0 + v2.0)
        dist, idx = self.tree.query(query_point, k=self.k)

        if dist[0] < 1e-12:
            return float(self.values[idx[0]])

        # Centralizamos y aplanamos para acceso rápido
        pts = self.points[idx] - query_point
        px = pts[:, 0]
        py = pts[:, 1]
        v = self.values[idx]

        EPS = 1e-14
        n = self.k

        # TRIPLE BUCLE OPTIMIZADO (v3.0)
        # Reemplazamos combinations para evitar overhead de tuplas y redundancia
        for i in range(n):
            xi, yi = px[i], py[i]
            vi = v[i]
            for j in range(i + 1, n):
                xj, yj = px[j], py[j]
                vj = v[j]

                # Pre-calculamos el producto cruzado ij (se usará n-j veces)
                cp_ij = xi * yj - yi * xj

                for k in range(j + 1, n):
                    xk, yk = px[k], py[k]

                    # Calculamos los otros dos productos cruzados
                    cp_jk = xj * yk - yj * xk
                    cp_ki = xk * yi - yk * xi

                    # EL ESCUDO (Mismo signo = origen encerrado)
                    if (cp_ij > -EPS and cp_jk > -EPS and cp_ki > -EPS) or \
                       (cp_ij <  EPS and cp_jk <  EPS and cp_ki <  EPS):

                        det = cp_ij + cp_jk + cp_ki
                        if abs(det) < 1e-18:
                            continue

                        # Retorno inmediato al encontrar el primer triángulo (el más cercano)
                        return float((cp_jk * vi + cp_ki * vj + cp_ij * v[k]) / det)

        return None

# Instrucción para el test:
# motor = FastSimplex2DAngular_Final3(k_neighbors=15)
# motor.fit(data_train)
 
