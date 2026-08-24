"""
Busqueda del circulo critico - Estabilidad de talud (Bishop simplificado)
----------------------------------------------------------------------------
Hasta ahora evaluamos un circulo de prueba fijo. Este modulo hace una
busqueda en grilla sobre (xc, yc, R) para encontrar el circulo que da
el FS MINIMO (el circulo critico) para cada uno de los 3 escenarios de
material/agua ya definidos (favorable, intermedio, desfavorable).

Esto es lo que en la practica se reporta: no el FS de un circulo
cualquiera, sino el FS del peor circulo encontrado.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from bishop_simplificado import (
    generar_dovelas,
    calcular_fs_bishop,
    calcular_fs_bishop_pseudoestatico,
    clasificar_fs,
    interpolar_terreno,
    elevacion_circulo,
)
from comparar_escenarios import perfil_talud, GAMMA, escenarios


def evaluar_circulo(xc, yc, R, perfil, c, phi_deg, gamma, ru, n_dovelas=15,
                     ancho_minimo=15.0, alpha_max_grados=60.0, kh=0.0, materiales=None,
                     elevacion_minima=None):
    """
    Intenta calcular el FS de un circulo (xc, yc, R). Retorna None si el
    circulo no es geometricamente valido o no es un candidato razonable:
      - no interseca el terreno, o produce dovelas con altura negativa
      - el momento motor neto no es positivo (direccion invalida)
      - la superficie es demasiado angosta (ancho_minimo): descarta
        "mordiscos" pequenos y poco profundos que no son deslizamientos
        representativos
      - alguna dovela tiene |alpha| mayor a alpha_max_grados: descarta
        superficies casi verticales donde las hipotesis de Bishop
        simplificado dejan de ser razonables
      - el punto mas bajo del circulo (yc - R) queda bajo elevacion_minima:
        descarta circulos que cruzarian una capa dura/roca impenetrable

    kh: coeficiente sismico horizontal. kh=0 (default) evalua la condicion
        estatica; kh>0 evalua la condicion pseudo-estatica sismica -- y en
        ese caso el circulo critico buscado puede ser distinto al circulo
        critico estatico, porque la fuerza horizontal cambia que geometria
        resulta "peor".

    materiales: opcional, para buscar el circulo critico en un talud de
        dos capas (ver generar_dovelas). Si se entrega, c/phi_deg/gamma
        solo se usan como valor por defecto (no deberian importar).

    elevacion_minima: opcional, elevacion de una capa dura/roca. Circulos
        cuyo punto mas bajo (yc - R) caiga bajo esta elevacion se descartan.
    """
    if elevacion_minima is not None and (yc - R) < elevacion_minima:
        return None

    try:
        dovelas = generar_dovelas(xc, yc, R, perfil, n_dovelas=n_dovelas, ru=ru,
                                   materiales=materiales)
    except ValueError:
        return None

    if len(dovelas) < 6 or any(d["h"] <= 0 for d in dovelas):
        return None

    ancho = dovelas[-1]["x_mid"] + dovelas[-1]["b"] / 2 - (dovelas[0]["x_mid"] - dovelas[0]["b"] / 2)
    if ancho < ancho_minimo:
        return None

    alpha_max_rad = math.radians(alpha_max_grados)
    if any(abs(d["alpha"]) > alpha_max_rad for d in dovelas):
        return None

    try:
        if kh > 0:
            fs, _ = calcular_fs_bishop_pseudoestatico(
                dovelas, c=c, phi_deg=phi_deg, gamma=gamma, yc=yc, R=R, kh=kh, max_iter=80
            )
        else:
            fs, _ = calcular_fs_bishop(dovelas, c=c, phi_deg=phi_deg, gamma=gamma, max_iter=80)
    except ValueError:
        return None

    return fs


def buscar_circulo_critico(perfil, c, phi_deg, gamma, ru,
                            xc_rango, yc_rango, R_rango, n_dovelas=15, kh=0.0,
                            materiales=None, elevacion_minima=None):
    """
    Busqueda en grilla: prueba todas las combinaciones (xc, yc, R) de los
    rangos dados y retorna el circulo con el FS minimo encontrado.
    """
    mejor = None

    for xc in xc_rango:
        for yc in yc_rango:
            for R in R_rango:
                fs = evaluar_circulo(xc, yc, R, perfil, c, phi_deg, gamma, ru,
                                      n_dovelas, kh=kh, materiales=materiales,
                                      elevacion_minima=elevacion_minima)
                if fs is None:
                    continue
                if mejor is None or fs < mejor["fs"]:
                    mejor = {"xc": xc, "yc": yc, "R": R, "fs": fs}

    if mejor is None:
        raise RuntimeError("No se encontro ningun circulo valido en el rango de busqueda.")

    return mejor


def graficar_circulo_individual(perfil, resultado):
    """
    Genera una figura independiente para UN escenario (talud + su circulo
    critico). Pensado para poder reutilizar cada imagen por separado en
    una app o pagina HTML (una tarjeta por escenario), en vez de un unico
    grafico combinado.
    """
    xc, yc, R = resultado["xc"], resultado["yc"], resultado["R"]
    fs = resultado["fs"]
    color = resultado["color"]
    nombre = resultado["nombre"]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    xs_terreno = [p[0] for p in perfil]
    ys_terreno = [p[1] for p in perfil]
    ax.plot(xs_terreno, ys_terreno, color="black", linewidth=2, label="Superficie del talud")

    theta = np.linspace(0, 2 * np.pi, 300)
    x_circ = xc + R * np.cos(theta)
    y_circ = yc + R * np.sin(theta)
    mascara = y_circ <= max(ys_terreno) + 10
    ax.plot(x_circ[mascara], y_circ[mascara], "--", color=color, linewidth=2,
            label="Superficie de falla critica")

    # Dovelas de la superficie critica, para dar contexto visual
    dovelas = generar_dovelas(xc, yc, R, perfil, n_dovelas=15, ru=resultado.get("ru", 0.0))
    for d in dovelas:
        x = d["x_mid"]
        y_terreno = interpolar_terreno(x, perfil)
        y_circulo = elevacion_circulo(x, xc, yc, R)
        ax.plot([x, x], [y_circulo, y_terreno], color="gray", linewidth=0.5, alpha=0.6)

    ax.plot(xc, yc, "+", color="black", markersize=10)

    ax.set_xlabel("Distancia horizontal (m)")
    ax.set_ylabel("Elevacion (m)")
    ax.set_title(f"{nombre} — FS critico = {fs:.2f} ({clasificar_fs(fs)})", color=color)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min(xs_terreno) - 5, max(xs_terreno) + 5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    nombre_archivo = (
        nombre.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(".", "")
    )
    ruta = f"/mnt/user-data/outputs/circulo_critico_{nombre_archivo}.png"
    plt.savefig(ruta, dpi=150)
    plt.show()
    plt.close(fig)
    return ruta


def graficar_circulos_criticos(perfil, resultados):
    """
    Genera UN grafico por escenario (uno por caso), en vez de un unico
    grafico combinado. Retorna la lista de rutas generadas, en el mismo
    orden que 'resultados' -- util si luego esto se muestra como cards
    en una app o pagina HTML.
    """
    rutas = []
    for r in resultados:
        ruta = graficar_circulo_individual(perfil, r)
        rutas.append(ruta)
    return rutas


if __name__ == "__main__":
    # Rango de busqueda (grilla). Mas fino = mas preciso pero mas lento.
    xc_rango = range(15, 75, 3)
    yc_rango = range(45, 110, 3)
    R_rango = range(35, 120, 3)

    KH_SISMICO = 0.15  # coeficiente sismico de referencia (DS248, zona media-alta)

    resultados_estatico = []
    resultados_sismico = []

    for nombre, datos in escenarios.items():
        print(f"--- Escenario '{nombre}' ---")

        mejor_estatico = buscar_circulo_critico(
            perfil_talud, c=datos["c"], phi_deg=datos["phi_deg"], gamma=GAMMA,
            ru=datos["ru"], xc_rango=xc_rango, yc_rango=yc_rango, R_rango=R_rango,
            kh=0.0,
        )
        mejor_estatico.update({"nombre": f"{nombre} (estatico)", "color": datos["color"], "ru": datos["ru"]})
        resultados_estatico.append(mejor_estatico)
        print(f"  Estatico   : xc={mejor_estatico['xc']}, yc={mejor_estatico['yc']}, "
              f"R={mejor_estatico['R']}  -> FS = {mejor_estatico['fs']:.3f} "
              f"({clasificar_fs(mejor_estatico['fs'])})")

        mejor_sismico = buscar_circulo_critico(
            perfil_talud, c=datos["c"], phi_deg=datos["phi_deg"], gamma=GAMMA,
            ru=datos["ru"], xc_rango=xc_rango, yc_rango=yc_rango, R_rango=R_rango,
            kh=KH_SISMICO,
        )
        mejor_sismico.update({"nombre": f"{nombre} (sismico kh={KH_SISMICO})",
                              "color": datos["color"], "ru": datos["ru"]})
        resultados_sismico.append(mejor_sismico)
        print(f"  Sismico    : xc={mejor_sismico['xc']}, yc={mejor_sismico['yc']}, "
              f"R={mejor_sismico['R']}  -> FS = {mejor_sismico['fs']:.3f} "
              f"({clasificar_fs(mejor_sismico['fs'])})")

        mismo_circulo = (mejor_estatico["xc"], mejor_estatico["yc"], mejor_estatico["R"]) == \
                        (mejor_sismico["xc"], mejor_sismico["yc"], mejor_sismico["R"])
        print(f"  Mismo circulo critico en ambas condiciones: {mismo_circulo}")

    # Un grafico por escenario y por condicion (6 imagenes en total)
    rutas = graficar_circulos_criticos(perfil_talud, resultados_estatico + resultados_sismico)
    for r, ruta in zip(resultados_estatico + resultados_sismico, rutas):
        print(f"Grafico de '{r['nombre']}' guardado en: {ruta}")
