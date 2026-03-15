
import numpy as np
import time
from scipy.spatial import cKDTree
from itertools import combinations

class FastSimplex2DAngular_Final2:
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
        # El árbol ya devuelve los índices ordenados por distancia de menor a mayor
        dist, idx = self.tree.query(query_point, k=self.k)

        # Caso 1: Coincidencia exacta (ahorro total de tiempo)
        if dist[0] < 1e-12:
            return float(self.values[idx[0]])

        # Centralizamos los puntos vecinos respecto a la consulta
        # pts[0] es el vecino más cercano, pts[14] el más lejano
        pts = self.points[idx] - query_point
        vals = self.values[idx]

        EPS = 1e-14

        # El iterador combinations(range(15), 3) genera:
        # (0,1,2), (0,1,3), (0,1,4)... es decir, SIEMPRE prioriza
        # las combinaciones de los puntos más cercanos.
        for i, j, k in combinations(range(len(idx)), 3):
            A, B, C = pts[i], pts[j], pts[k]

            # Productos cruzados (2x2 determinants)
            # Representan el doble del área de los sub-triángulos formados con el origen
            cp1 = A[0]*B[1] - A[1]*B[0]
            cp2 = B[0]*C[1] - B[1]*C[0]
            cp3 = C[0]*A[1] - C[1]*A[0]

            # DETERMINACIÓN DE ENCIERRO:
            # Si los tres productos tienen el mismo signo, el origen está dentro.
            if (cp1 > -EPS and cp2 > -EPS and cp3 > -EPS) or \
               (cp1 <  EPS and cp2 <  EPS and cp3 <  EPS):

                # El determinante total es la suma de las áreas parciales
                det = cp1 + cp2 + cp3
                if abs(det) < 1e-18:
                    continue

                # CIERRE INMEDIATO: Al encontrar el primer triángulo que encierra,
                # salimos del bucle. Como probamos por cercanía, este triángulo
                # es el más óptimo localmente.
                return float((cp2*vals[i] + cp3*vals[j] + cp1*vals[k]) / det)

        # Si tras 455 combinaciones no hay encierro, devolvemos None (Extrapolación no permitida)
        return None
       

# El test de ChatGPT (100k puntos / 100k consultas) se corre usando esta clase.
