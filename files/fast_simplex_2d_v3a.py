
import numpy as np
import time
from scipy.spatial import cKDTree
from itertools import combinations

class FastSimplex2DAngular_Final:
    def __init__(self, k_neighbors=15):
        self.k = k_neighbors
        self.tree = None
        self.points = None
        self.values = None

    def fit(self, data):
        # data: (N, 3) -> [x, y, valor]
        self.points = data[:, :2].copy()
        self.values = data[:, 2].copy()
        self.tree = cKDTree(self.points)
        return self

    def predict(self, query_point):
        dist, idx = self.tree.query(query_point, k=self.k)
        if dist[0] < 1e-12: return float(self.values[idx[0]])

        # Centralizamos y preparamos valores locales
        pts = self.points[idx] - query_point
        vals = self.values[idx] # Valores de los 15 vecinos

        # Tolerancia para error numérico y "extrapolación casi-interp"
        EPS = 1e-14

        # Buscamos el triángulo que encierra al origen
        for i, j, k in combinations(range(len(idx)), 3):
            A, B, C = pts[i], pts[j], pts[k]

            # Productos cruzados (áreas orientadas)
            cp1 = A[0]*B[1] - A[1]*B[0]
            cp2 = B[0]*C[1] - B[1]*C[0]
            cp3 = C[0]*A[1] - C[1]*A[0]

            # EL ESCUDO: Verificamos si el origen está dentro (mismo signo o casi 0)
            if (cp1 > -EPS and cp2 > -EPS and cp3 > -EPS) or \
               (cp1 <  EPS and cp2 <  EPS and cp3 <  EPS):

                det = cp1 + cp2 + cp3
                if abs(det) < 1e-18: continue

                # Interpolación: Usamos 'vals' que ya tiene el orden de los vecinos
                return float((cp2*vals[i] + cp3*vals[j] + cp1*vals[k]) / det)

        return None

# --- TEST 2D DEFINITIVO ---
N_POINTS = 5000
N_QUERIES = 1000
np.random.seed(42)

data_train = np.random.rand(N_POINTS, 3)
data_train[:, 2] = np.sin(data_train[:, 0]*5) * np.cos(data_train[:, 1]*5)
queries = np.random.rand(N_QUERIES, 2)

motor = FastSimplex2DAngular_Final(k_neighbors=15)
motor.fit(data_train)

t0 = time.time()
res = [motor.predict(q) for q in queries]
tt = time.time() - t0
acc = sum(1 for r in res if r is not None) / N_QUERIES * 100

print(f"======= MOTOR 2D FINAL: EQUIPO EDA =======")
print(f"Aciertos: {acc:.2f}%")
print(f"Tiempo:   {tt:.4f}s")
print(f"===========================================")
 
