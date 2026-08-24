"""
Suite de tests - Estabilidad de talud
--------------------------------------
Correr con:
    pytest test_bishop.py -v

Cubre tres cosas:
1. Regresion: valores ya conocidos que no deben cambiar (protege contra
   romper algo sin darse cuenta al seguir agregando features).
2. Casos limite: geometrias invalidas, direcciones de circulo incorrectas.
3. Validacion contra benchmarks publicados (ACADS, Loukidis) -- version
   automatizada de validacion_acads.py / validacion_loukidis.py, para que
   corran junto con el resto de los tests en vez de a mano.
"""

import math
import pytest

from bishop_simplificado import (
    generar_dovelas,
    calcular_fs_bishop,
    calcular_fs_bishop_pseudoestatico,
)
from busqueda_circulo_critico import buscar_circulo_critico, evaluar_circulo
from comparar_escenarios import perfil_talud, GAMMA


# ------------------------------------------------------------------
# 1. Regresion -- valores conocidos que no deben cambiar
# ------------------------------------------------------------------

def test_fs_basico_talud_favorable():
    """Circulo y escenario 'Favorable' ya validado en el proyecto: FS ~ 1.099."""
    xc, yc, R = 72, 66, 65
    dovelas = generar_dovelas(xc, yc, R, perfil_talud, n_dovelas=20, ru=0.0)
    fs, _ = calcular_fs_bishop(dovelas, c=5.0, phi_deg=30.0, gamma=GAMMA)
    assert fs == pytest.approx(1.099, abs=0.01)


def test_fs_pseudoestatico_kh0_igual_a_estatico():
    """kh=0 en el solver sismico debe coincidir exactamente con el estatico."""
    xc, yc, R = 72, 66, 65
    dovelas = generar_dovelas(xc, yc, R, perfil_talud, n_dovelas=20, ru=0.0)
    fs_estatico, _ = calcular_fs_bishop(dovelas, c=5.0, phi_deg=30.0, gamma=GAMMA)
    fs_pseudo, _ = calcular_fs_bishop_pseudoestatico(
        dovelas, c=5.0, phi_deg=30.0, gamma=GAMMA, yc=yc, R=R, kh=0.0
    )
    assert fs_estatico == pytest.approx(fs_pseudo, abs=1e-6)


def test_fs_pseudoestatico_kh_alto_reduce_fs():
    """Mas sismo siempre debe dar un FS menor o igual (nunca mas seguro)."""
    xc, yc, R = 72, 66, 65
    dovelas = generar_dovelas(xc, yc, R, perfil_talud, n_dovelas=20, ru=0.0)
    fs_bajo, _ = calcular_fs_bishop_pseudoestatico(
        dovelas, c=5.0, phi_deg=30.0, gamma=GAMMA, yc=yc, R=R, kh=0.05
    )
    fs_alto, _ = calcular_fs_bishop_pseudoestatico(
        dovelas, c=5.0, phi_deg=30.0, gamma=GAMMA, yc=yc, R=R, kh=0.15
    )
    assert fs_alto < fs_bajo


def test_formato_capas_equivale_a_formato_materiales_antiguo():
    """El nuevo formato 'capas' (N capas) debe reproducir EXACTO el
    resultado del formato antiguo 'materiales' (2 capas) para el mismo caso."""
    xc, yc, R = 72, 66, 65
    contacto = [(x, y - 3) for x, y in perfil_talud]

    materiales = {
        "contacto": contacto,
        "superior": {"c": 5.0, "phi_deg": 30.0, "gamma": 18.0},
        "inferior": {"c": 20.0, "phi_deg": 35.0, "gamma": 20.0},
    }
    dovelas_viejo = generar_dovelas(xc, yc, R, perfil_talud, n_dovelas=20, materiales=materiales)
    fs_viejo, _ = calcular_fs_bishop(dovelas_viejo, c=0, phi_deg=0, gamma=0)

    capas = [
        {"piso": contacto, "c": 5.0, "phi_deg": 30.0, "gamma": 18.0},
        {"piso": None, "c": 20.0, "phi_deg": 35.0, "gamma": 20.0},
    ]
    dovelas_nuevo = generar_dovelas(xc, yc, R, perfil_talud, n_dovelas=20, capas=capas)
    fs_nuevo, _ = calcular_fs_bishop(dovelas_nuevo, c=0, phi_deg=0, gamma=0)

    assert fs_viejo == pytest.approx(fs_nuevo, abs=1e-9)


def test_tres_capas_da_fs_intermedio():
    """Con 3 capas (debil / media / fuerte), el FS debe quedar entre el
    caso 'todo material debil' y el caso 'todo material fuerte'."""
    xc, yc, R = 72, 66, 65
    contacto1 = [(x, y - 2) for x, y in perfil_talud]
    contacto2 = [(x, y - 6) for x, y in perfil_talud]

    capas_tres = [
        {"piso": contacto1, "c": 2.0, "phi_deg": 25.0, "gamma": 16.0},
        {"piso": contacto2, "c": 8.0, "phi_deg": 30.0, "gamma": 18.0},
        {"piso": None, "c": 20.0, "phi_deg": 35.0, "gamma": 20.0},
    ]
    dovelas = generar_dovelas(xc, yc, R, perfil_talud, n_dovelas=20, capas=capas_tres)
    fs_tres, _ = calcular_fs_bishop(dovelas, c=0, phi_deg=0, gamma=0)

    dovelas_debil = generar_dovelas(xc, yc, R, perfil_talud, n_dovelas=20)
    fs_debil, _ = calcular_fs_bishop(dovelas_debil, c=2.0, phi_deg=25.0, gamma=16.0)

    dovelas_fuerte = generar_dovelas(xc, yc, R, perfil_talud, n_dovelas=20)
    fs_fuerte, _ = calcular_fs_bishop(dovelas_fuerte, c=20.0, phi_deg=35.0, gamma=20.0)

    assert fs_debil < fs_tres < fs_fuerte


# ------------------------------------------------------------------
# 2. Casos limite
# ------------------------------------------------------------------

def test_circulo_que_no_interseca_el_perfil_lanza_error():
    with pytest.raises(ValueError):
        generar_dovelas(xc=5, yc=5, R=1, perfil=perfil_talud, n_dovelas=10)


def test_circulo_con_direccion_invalida_lanza_error_en_solver():
    """Un circulo cuyo centro esta 'del lado equivocado' debe fallar de
    forma controlada (ValueError), no devolver un FS sin sentido."""
    xc, yc, R = 20, 35, 22  # circulo con momento motor neto negativo (visto en el proyecto)
    dovelas = generar_dovelas(xc, yc, R, perfil_talud, n_dovelas=15, ru=0.2)
    with pytest.raises(ValueError):
        calcular_fs_bishop(dovelas, c=5.0, phi_deg=30.0, gamma=GAMMA)


# ------------------------------------------------------------------
# 3. Validacion contra benchmarks publicados
# ------------------------------------------------------------------

def test_benchmark_acads_simple_slope():
    """
    Giam & Donald (1989) / GeoStudio SLOPE/W verification manual.
    FS publicado: ACADS=1.00, SLOPE/W Bishop=0.963.
    Se acepta un margen de +-5% para considerar la validacion exitosa.
    """
    perfil = [(20, 35), (40, 35), (60, 25), (70, 25)]  # perfil espejado
    mejor = buscar_circulo_critico(
        perfil, c=3.0, phi_deg=19.6, gamma=20.0, ru=0.0,
        xc_rango=range(50, 70, 2), yc_rango=range(42, 62, 2), R_rango=range(15, 35, 2),
        n_dovelas=25,
    )
    assert mejor["fs"] == pytest.approx(1.00, abs=0.05)


def test_benchmark_loukidis_coeficiente_sismico_critico():
    """
    Loukidis, Bandini & Salgado (2005). kc publicado (Bishop simplificado)
    para beta=30, phi=20, lambda=0.137: kc=0.114 (debe dar FS~1.0 en ese kh).
    """
    gamma, H, phi_deg, lam = 20.0, 20.0, 20.0, 0.137
    c = lam * gamma * H * math.tan(math.radians(phi_deg))
    beta_deg, D = 30.0, 1.5
    run_pendiente = H / math.tan(math.radians(beta_deg))
    y_crest = 30.0
    y_toe = y_crest - H
    y_hard = y_crest - D * H
    ext = 40.0
    perfil = [(0, y_crest), (ext, y_crest),
              (ext + run_pendiente, y_toe), (ext + run_pendiente + ext, y_toe)]

    mejor_fs = None
    for xc in range(50, 95, 2):
        for yc in range(35, 80, 2):
            for R in range(25, 70, 2):
                fs = evaluar_circulo(
                    xc, yc, R, perfil, c=c, phi_deg=phi_deg, gamma=gamma, ru=0.0,
                    n_dovelas=20, kh=0.114, elevacion_minima=y_hard,
                    alpha_max_grados=89, ancho_minimo=5,
                )
                if fs is not None and (mejor_fs is None or fs < mejor_fs):
                    mejor_fs = fs

    assert mejor_fs == pytest.approx(1.0, abs=0.03)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
