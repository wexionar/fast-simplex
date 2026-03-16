
import time
import numpy as np
from scipy.spatial import cKDTree

# --- MOTOR 1: LUMIN EMBUDO 3D (BORRADOR) ---
class LuminEmbudo3D:
    def __init__(self, data):
        self.X = np.array(data[:, :-1], dtype=np.float64)
        self.Y = np.array(data[:, -1], dtype=np.float64)
        self.tree = cKDTree(self.X)
        self.d = 3

    def predict(self, p, lambda_coef=15):
        p = np.asarray(p, dtype=np.float64)
        k = self.d * lambda_coef
        dists, idxs = self.tree.query(p, k=min(k, len(self.X)))
        v1_idx = idxs[0]
        v1_orig = self.X[v1_idx]
        vec_v1 = v1_orig - p
        dist_v1 = np.linalg.norm(vec_v1)
        if dist_v1 < 1e-12: return self.Y[v1_idx]

        u_x = vec_v1 / dist_v1
        v_temp = np.array([1, 0, 0]) if abs(u_x[0]) < 0.9 else np.array([0, 1, 0])
        u_y = np.cross(u_x, v_temp)
        u_y /= np.linalg.norm(u_y)
        u_z = np.cross(u_x, u_y)

        v_locales, v_indices = [], []
        for i in idxs[1:]:
            diff = self.X[i] - p
            vx = np.dot(diff, u_x)
            if vx < 0:
                v_locales.append([np.dot(diff, u_y), np.dot(diff, u_z)])
                v_indices.append(i)

        if len(v_locales) < 3: return None

        v_locales = np.array(v_locales)
        v2_rel_idx = 0
        u_x_2d = v_locales[v2_rel_idx] / np.linalg.norm(v_locales[v2_rel_idx])
        u_y_2d = np.array([-u_x_2d[1], u_x_2d[0]])

        v3_idx_final, v4_idx_final = None, None
        signo_y_ref = 0
        for i in range(1, len(v_locales)):
            diff_2d = v_locales[i]
            vx_2d = np.dot(diff_2d, u_x_2d)
            vy_2d = np.dot(diff_2d, u_y_2d)
            if vx_2d < 0:
                if v3_idx_final is None:
                    v3_idx_final = i
                    signo_y_ref = np.sign(vy_2d)
                elif np.sign(vy_2d) != signo_y_ref and abs(vy_2d) > 1e-12:
                    v4_idx_final = i
                    break

        if v4_idx_final is None: return None

        final_idxs = [v1_idx, v_indices[v2_rel_idx], v_indices[v3_idx_final], v_indices[v4_idx_final]]
        sx, sy = self.X[final_idxs], self.Y[final_idxs]
        try:
            T = (sx[1:] - sx[0]).T
            w = np.linalg.solve(T, p - sx[0])
            pesos = np.append(1.0 - np.sum(w), w)
            return np.dot(pesos, sy) if np.all(pesos >= -1e-10) else None
        except: return None

# --- MOTOR 2: ANGULAR V3 (TU NUEVA LÓGICA) ---
class FastSimplex3DAngular_V3:
    def __init__(self, data, k_neighbors=15):
        self.data = data
        self.tree = cKDTree(data[:, :3])
        self.k = k_neighbors

    def predict(self, q):
        dist, idx = self.tree.query(q, k=self.k)
        pts = self.data[idx, :3] - q
        vals = self.data[idx, 3]

        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    A, B, C = pts[i], pts[j], pts[k]
                    normal = np.cross(B - A, C - A)
                    dist_origen = np.dot(A, normal)
                    if abs(dist_origen) < 1e-12: continue

                    for l in range(k + 1, n):
                        D = pts[l]
                        if dist_origen * np.dot(D - A, normal) < -1e-12:
                            # Candidato encontrado, validamos baricéntricas
                            M = np.vstack([np.array([A, B, C, D]).T, [1, 1, 1, 1]])
                            try:
                                w = np.linalg.solve(M, [0, 0, 0, 1])
                                if np.all(w >= -1e-10):
                                    return np.dot(w, vals[idx[[i, j, k, l]]])
                            except: continue
        return None

# --- BENCHMARK ---
def run_comparison():
    N_POINTS, N_QUERIES = 10000, 1000
    pts = np.random.rand(N_POINTS, 3) * 10
    vals = np.sin(pts[:, 0]) + np.cos(pts[:, 1]) + np.sin(pts[:, 2])
    data = np.column_stack([pts, vals])
    queries = np.random.rand(N_QUERIES, 3) * 10

    # Test Embudo v5.0
    engine_v5 = LuminEmbudo3D(data)
    t0 = time.time()
    res_v5 = [engine_v5.predict(q) for q in queries]
    t_v5 = time.time() - t0
    valid_v5 = sum(1 for r in res_v5 if r is not None)

    # Test Angular V3
    engine_v3 = FastSimplex3DAngular_V3(data)
    t1 = time.time()
    res_v3 = [engine_v3.predict(q) for q in queries]
    t_v3 = time.time() - t1
    valid_v3 = sum(1 for r in res_v3 if r is not None)

    print(f"--- RESULTADOS 3D ---")
    print(f"Embudo v5.0:  Tiempo {t_v5:.4f}s | Cobertura {valid_v5/N_QUERIES*100:.2f}%")
    print(f"Angular V3:   Tiempo {t_v3:.4f}s | Cobertura {valid_v3/N_QUERIES*100:.2f}%")

if __name__ == "__main__":
    run_comparison()
  
