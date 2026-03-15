
# version 3D version 5.0
import numpy as np
from scipy.spatial import cKDTree

class LuminEmbudo3D:
    def __init__(self, data):
        # data: [X, Y, Z, Valor]
        self.X = np.array(data[:, :-1], dtype=np.float64)
        self.Y = np.array(data[:, -1], dtype=np.float64)
        self.tree = cKDTree(self.X)
        self.d = 3

    def predict(self, p, lambda_coef=15): # Más vecinos en 3D para asegurar cierre
        p = np.asarray(p, dtype=np.float64)
        k = self.d * lambda_coef
        dists, idxs = self.tree.query(p, k=min(k, len(self.X)))

        # 1. EL ANCLA (V1): El más cercano define el Eje X
        v1_idx = idxs[0]
        v1_orig = self.X[v1_idx]
        vec_v1 = v1_orig - p
        dist_v1 = np.linalg.norm(vec_v1)

        if dist_v1 < 1e-12: return self.Y[v1_idx], "Éxito: Coincidencia exacta"

        # --- SISTEMA LOCAL 3D ---
        # Definimos u_x apuntando a V1
        u_x = vec_v1 / dist_v1

        # Generamos una base ortonormal (u_x, u_y, u_z)
        # Buscamos un vector no colineal para Gram-Schmidt
        v_temp = np.array([1, 0, 0]) if abs(u_x[0]) < 0.9 else np.array([0, 1, 0])
        u_y = np.cross(u_x, v_temp)
        u_y /= np.linalg.norm(u_y)
        u_z = np.cross(u_x, u_y)

        # 2. FILTRO "MURO FRONTAL" (x > 0 respecto a Pc)
        v_locales = []
        v_indices = []

        for i in idxs[1:]:
            diff = self.X[i] - p
            vx = np.dot(diff, u_x)
            vy = np.dot(diff, u_y)
            vz = np.dot(diff, u_z)

            # Si vx < 0, está del lado opuesto a V1 (adelante de Pc)
            if vx < 0:
                v_locales.append([vy, vz]) # Proyectamos a 2D (plano YZ)
                v_indices.append(i)

        if len(v_locales) < 3:
            return None, "Soporte insuficiente: pocos puntos adelante"

        # 3. EL CIERRE 2D (Embudo sobre plano YZ)
        # Ahora buscamos V2, V3, V4 que encierren (0,0) en el plano YZ
        v_locales = np.array(v_locales)

        # Aplicamos nuestra lógica de signos de la v4.5 en el plano YZ
        # v_locales[0] es el nuevo "v1" del plano 2D
        v2_rel_idx = 0
        y_v2, z_v2 = v_locales[v2_rel_idx]
        dist_v2_2d = np.linalg.norm(v_locales[v2_rel_idx])

        # Buscamos v3 y v4 con signos opuestos en el nuevo eje local del plano
        u_x_2d = v_locales[v2_rel_idx] / dist_v2_2d
        u_y_2d = np.array([-u_x_2d[1], u_x_2d[0]])

        v3_idx_final, v4_idx_final = None, None
        signo_y_ref = 0

        for i in range(1, len(v_locales)):
            diff_2d = v_locales[i]
            vx_2d = np.dot(diff_2d, u_x_2d)
            vy_2d = np.dot(diff_2d, u_y_2d)

            if vx_2d < 0: # Lado opuesto al V2 proyectado
                if v3_idx_final is None:
                    v3_idx_final = i
                    signo_y_ref = np.sign(vy_2d)
                elif np.sign(vy_2d) != signo_y_ref and abs(vy_2d) > 1e-12:
                    v4_idx_final = i
                    break

        if v4_idx_final is None:
            return None, "Soporte insuficiente: no se cierra el tetraedro"

        # 4. CÁLCULO FINAL (Baricéntricas 3D)
        final_idxs = [v1_idx, v_indices[v2_rel_idx], v_indices[v3_idx_final], v_indices[v4_idx_final]]
        sx, sy = self.X[final_idxs], self.Y[final_idxs]

        # Resolver sistema para pesos baricéntricos en 3D
        try:
            # Matriz T: [V2-V1, V3-V1, V4-V1]
            T = (sx[1:] - sx[0]).T
            w = np.linalg.solve(T, p - sx[0])
            pesos = np.append(1.0 - np.sum(w), w)

            if np.all(pesos >= -1e-10):
                return np.dot(pesos, sy), "Éxito: Tetraedro Embudo v5.0"
            else:
                return None, "Punto fuera del tetraedro"
        except:
            return None, "Error: Simplex degenerado"

# --- TEST 3D ---
# Vamos a crear una esfera de puntos rodeando al origen (0,0,0)
data_3d = np.array([
    [-1,  0,  0, 100], # V1: Atrás
    [ 1,  1,  0, 200], # V2: Adelante-Arriba
    [ 1, -1,  1, 300], # V3: Adelante-Abajo-Derecha
    [ 1, -1, -1, 400], # V4: Adelante-Abajo-Izquierda
])

engine = LuminEmbudo3D(data_3d)
res, msg = engine.predict([0, 0, 0])
print(f"Resultado: {res} | {msg}")
 
