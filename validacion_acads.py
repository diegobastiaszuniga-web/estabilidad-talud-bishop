"""
Validacion contra benchmark publicado - ACADS Simple Slope
--------------------------------------------------------------
Caso de verificacion oficial publicado en:
    Giam, P.S.K. and Donald, I.B. (1989), "Example problems for testing
    soil slope stability programs", Monash University.
Y reproducido en el manual de verificacion de GeoStudio SLOPE/W (2022).

Geometria original (segun figura del manual, coordenadas en metros):
    (20,25) -> (30,25) -> (50,35) -> (70,35)

Nuestra convencion de signo para alpha asume que el talud desciende hacia
+x, así que se refleja horizontalmente (x' = 90 - x) para obtener una
geometria fisicamente identica que sí calza con esa convencion.

Valores publicados de FS:
    ACADS (estudio original): 1.00
    SLOPE/W, metodo de Bishop: 0.963
"""

from busqueda_circulo_critico import buscar_circulo_critico
from bishop_simplificado import clasificar_fs

PERFIL_ACADS = [(20, 35), (40, 35), (60, 25), (70, 25)]  # ya espejado
C = 3.0        # kPa
PHI_DEG = 19.6  # grados
GAMMA = 20.0    # kN/m3

FS_PUBLICADO_ACADS = 1.00
FS_PUBLICADO_SLOPEW_BISHOP = 0.963


def validar():
    mejor = buscar_circulo_critico(
        PERFIL_ACADS, c=C, phi_deg=PHI_DEG, gamma=GAMMA, ru=0.0,
        xc_rango=range(50, 70, 1),
        yc_rango=range(42, 62, 1),
        R_rango=range(15, 35, 1),
        n_dovelas=25,
    )

    fs_propio = mejor["fs"]
    diff_acads = 100 * (fs_propio - FS_PUBLICADO_ACADS) / FS_PUBLICADO_ACADS
    diff_slopew = 100 * (fs_propio - FS_PUBLICADO_SLOPEW_BISHOP) / FS_PUBLICADO_SLOPEW_BISHOP

    print(f"Circulo critico encontrado: xc={mejor['xc']}, yc={mejor['yc']}, R={mejor['R']}")
    print(f"FS (nuestro codigo):        {fs_propio:.3f}  ({clasificar_fs(fs_propio)})")
    print(f"FS publicado (ACADS):       {FS_PUBLICADO_ACADS:.3f}  (diferencia: {diff_acads:+.1f}%)")
    print(f"FS publicado (SLOPE/W):     {FS_PUBLICADO_SLOPEW_BISHOP:.3f}  (diferencia: {diff_slopew:+.1f}%)")


if __name__ == "__main__":
    validar()
