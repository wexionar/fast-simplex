
import numpy as np
from scipy.spatial import cKDTree

class FastSimplex2D5:
    def __init__(self, k_neighbors=15):
        self.k = k_neighbors
        self.tree = None
        self.points = None
        self.values = None

    def fit(self, data):
        self.points = np.ascontiguousarray(data[:, :2], dtype=np.float64)
        self.values = np.ascontiguousarray(data[:, 2], dtype=np.float64)
        self.tree = cKDTree(self.points)
        return self

    def predict(self, query_point):
        # 1. Obtenemos los K más cercanos (ya vienen ordenados por distancia)
        dist, idx = self.tree.query(query_point, k=self.k)
        if dist[0] < 1e-12: return float(self.values[idx[0]])

        # 2. Coordenadas locales respecto al punto de consulta
        pts = self.points[idx] - query_point
        px, py = pts[:, 0], pts[:, 1]
        v = self.values[idx]

        EPS = 1e-15
        n = self.k

        # 3. Triple bucle de búsqueda inmediata
        # Al estar los puntos ordenados por el árbol, el primer triángulo
        # que encontremos será, por definición, el más PRÓXIMO.
        for i in range(n):
            xi, yi, vi = px[i], py[i], v[i]
            for j in range(i + 1, n):
                xj, yj, vj = px[j], py[j], v[j]
                cp_ij = xi * yj - yi * xj

                # Optimizamos: solo buscamos el tercero si i y j no están alineados con el origen
                if abs(cp_ij) < 1e-18: continue

                for k in range(j + 1, n):
                    cp_jk = xj * py[k] - yj * px[k]
                    cp_ki = px[k] * yi - py[k] * xi

                    # EL ESCUDO: Verificamos si el origen (0,0) está encerrado
                    if (cp_ij > -EPS and cp_jk > -EPS and cp_ki > -EPS) or \
                       (cp_ij <  EPS and cp_jk <  EPS and cp_ki <  EPS):

                        det = cp_ij + cp_jk + cp_ki
                        # Si encontramos el triángulo más cercano, salimos volando.
                        # La precisión la da el hecho de que i, j, k son los índices más bajos posibles.
                        return float((cp_jk * vi + cp_ki * vj + cp_ij * v[k]) / det)

        return None
       
