"""
Validacion contra benchmark publicado - Loukidis, Bandini & Salgado (2005/2006)
--------------------------------------------------------------------------------
Caso de verificacion sismico: coeficiente sismico critico kc para un talud
homogeneo con capa dura (roca) a una profundidad D*H bajo la cresta.

Fuente:
    Loukidis, D., Bandini, P. and Salgado, R. (2005/2006). "Critical seismic
    coefficient using limit analysis and finite elements". Proc. 16th Intl.
    Conf. on Soil Mechanics and Geotechnical Engineering, pp. 2685-2688.
    Tabla 1 (beta=30 grados): kc (Bishop simplificado) = 0.114 para
    phi'=20 grados, lambda=c/(gamma*H*tanphi')=0.137.

Los parametros originales (gamma=20 kN/m3, H=20m) se usan directamente,
ya que el paper los reporta explicitamente (aunque el resultado es
adimensional via lambda).
"""

import math
from busqueda_circulo_critico import evaluar_circulo

# --- Geometria reconstruida del caso Loukidis ---
GAMMA = 20.0    # kN/m3
H = 20.0        # m
PHI_DEG = 20.0
LAMBDA = 0.137
C = LAMBDA * GAMMA * H * math.tan(math.radians(PHI_DEG))  # ~19.95 kPa

BETA_DEG = 30.0
D = 1.5  # razon profundidad capa dura / altura del talud

EXT = 40.0  # largo de las extensiones planas (cresta y pie)
RUN_PENDIENTE = H / math.tan(math.radians(BETA_DEG))
Y_CREST = 30.0
Y_TOE = Y_CREST - H
Y_HARD = Y_CREST - D * H  # elevacion de la capa dura/roca

PERFIL = [
    (0, Y_CREST),
    (EXT, Y_CREST),
    (EXT + RUN_PENDIENTE, Y_TOE),
    (EXT + RUN_PENDIENTE + EXT, Y_TOE),
]

KC_PUBLICADO_BISHOP = 0.114


def fs_critico(kh, xc_rango, yc_rango, R_rango, n_dovelas=20):
    """FS minimo (circulo critico) para un kh dado, respetando la capa dura."""
    mejor = None
    for xc in xc_rango:
        for yc in yc_rango:
            for R in R_rango:
                fs = evaluar_circulo(
                    xc, yc, R, PERFIL, c=C, phi_deg=PHI_DEG, gamma=GAMMA, ru=0.0,
                    n_dovelas=n_dovelas, kh=kh, elevacion_minima=Y_HARD,
                    alpha_max_grados=89, ancho_minimo=5,
                )
                if fs is None:
                    continue
                if mejor is None or fs < mejor:
                    mejor = fs
    return mejor


def validar():
    print(f"Geometria: beta={BETA_DEG} grados, phi'={PHI_DEG} grados, "
          f"c'={C:.2f} kPa, gamma={GAMMA} kN/m3, capa dura en y={Y_HARD}")
    print()

    # Grilla acotada alrededor de la zona donde ya sabemos que esta el optimo
    # (una busqueda amplia inicial identifico esta region; se acota aqui para
    # que la validacion corra rapido)
    xc_rango = range(50, 95, 2)
    yc_rango = range(35, 80, 2)
    R_rango = range(25, 70, 2)

    fs_en_kc_publicado = fs_critico(KC_PUBLICADO_BISHOP, xc_rango, yc_rango, R_rango)

    print(f"FS con kh = kc publicado ({KC_PUBLICADO_BISHOP}): {fs_en_kc_publicado:.4f}")
    print("(deberia ser ~1.000 si nuestro codigo reproduce el mismo kc critico)")
    diff = 100 * (fs_en_kc_publicado - 1.0)
    print(f"Diferencia respecto a FS=1.0: {diff:+.2f}%")


if __name__ == "__main__":
    validar()
