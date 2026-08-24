"""
Comparacion de escenarios - Estabilidad de talud (Bishop simplificado)
-----------------------------------------------------------------------
Usa la misma geometria de talud y el mismo circulo de prueba del modulo
anterior, pero varia las condiciones del material/agua para mostrar
como se mueve el FS entre un escenario favorable, uno intermedio y uno
desfavorable. Esto es tipico en un informe de estabilidad: no se reporta
un solo FS, sino un rango segun las condiciones esperadas en terreno.
"""

import math
import matplotlib.pyplot as plt

from bishop_simplificado import (
    generar_dovelas,
    calcular_fs_bishop,
    clasificar_fs,
    interpolar_terreno,
    elevacion_circulo,
)

# ------------------------------------------------------------------
# Geometria (igual que el modulo anterior)
# ------------------------------------------------------------------

perfil_talud = [
    (0, 30),
    (20, 30),
    (65, 0),
    (100, 0),
]

xc, yc, R = 40, 75, 55
GAMMA = 18.0  # kN/m3, peso unitario del material (igual en los 3 escenarios)

# ------------------------------------------------------------------
# Definicion de los 3 escenarios
# ------------------------------------------------------------------
# ru: coeficiente de presion de poros (0 = talud drenado, valores altos
#     = nivel freatico alto / relave saturado)
# c, phi: resistencia al corte del material. Se reducen en el escenario
#     desfavorable para simular perdida de resistencia (ej. saturacion,
#     remoldeo, o un evento sismico que ademas reduce phi movilizado).

escenarios = {
    "Favorable": {
        "descripcion": "Talud bien drenado, material competente",
        "ru": 0.0,
        "c": 5.0,
        "phi_deg": 30.0,
        "color": "#0F6E56",  # verde/teal oscuro
    },
    "Intermedio": {
        "descripcion": "Nivel freatico parcial, condicion tipica de diseno",
        "ru": 0.3,
        "c": 5.0,
        "phi_deg": 30.0,
        "color": "#854F0B",  # amber oscuro
    },
    "Desfavorable": {
        "descripcion": "Relave saturado, perdida de resistencia por poros altos",
        "ru": 0.5,
        "c": 2.0,
        "phi_deg": 27.0,
        "color": "#791F1F",  # rojo oscuro
    },
}


def evaluar_escenario(nombre, datos):
    dovelas = generar_dovelas(xc, yc, R, perfil_talud, n_dovelas=20, ru=datos["ru"])
    fs, historial = calcular_fs_bishop(
        dovelas, c=datos["c"], phi_deg=datos["phi_deg"], gamma=GAMMA
    )
    return {
        "nombre": nombre,
        "fs": fs,
        "iteraciones": len(historial) - 1,
        **datos,
    }


def graficar_comparacion(resultados):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [1.1, 1]})

    # --- Panel 1: seccion del talud con el circulo comun a los 3 casos ---
    xs_terreno = [p[0] for p in perfil_talud]
    ys_terreno = [p[1] for p in perfil_talud]
    ax1.plot(xs_terreno, ys_terreno, color="black", linewidth=2, label="Superficie del talud")

    import numpy as np
    theta = np.linspace(0, 2 * np.pi, 300)
    x_circ = xc + R * np.cos(theta)
    y_circ = yc + R * np.sin(theta)
    mascara = y_circ <= max(ys_terreno) + 8
    ax1.plot(x_circ[mascara], y_circ[mascara], "--", color="gray",
              linewidth=1.5, label="Superficie de falla (misma para los 3 casos)")

    ax1.set_xlabel("Distancia horizontal (m)")
    ax1.set_ylabel("Elevacion (m)")
    ax1.set_title("Geometria comun evaluada")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: barras de FS por escenario ---
    nombres = [r["nombre"] for r in resultados]
    valores_fs = [r["fs"] for r in resultados]
    colores = [r["color"] for r in resultados]

    barras = ax2.bar(nombres, valores_fs, color=colores, width=0.55)
    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="FS = 1.0 (falla teorica)")
    ax2.axhline(y=1.5, color="gray", linestyle=":", linewidth=1, label="FS = 1.5 (referencia estatica)")

    for barra, r in zip(barras, resultados):
        ax2.text(barra.get_x() + barra.get_width() / 2, r["fs"] + 0.03,
                  f"{r['fs']:.2f}", ha="center", fontsize=11, fontweight="bold")
        ax2.text(barra.get_x() + barra.get_width() / 2, 0.05,
                  clasificar_fs(r["fs"]), ha="center", fontsize=8,
                  color="white", rotation=90, va="bottom")

    ax2.set_ylabel("Factor de Seguridad (FS)")
    ax2.set_title("FS segun condicion de poros y resistencia")
    ax2.set_ylim(0, max(valores_fs) * 1.25)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/comparacion_escenarios.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    resultados = []
    for nombre, datos in escenarios.items():
        r = evaluar_escenario(nombre, datos)
        resultados.append(r)
        print(f"{nombre:12s} | ru={datos['ru']:.2f}  c={datos['c']:.1f} kPa  "
              f"phi={datos['phi_deg']:.1f} deg  ->  FS = {r['fs']:.3f} "
              f"({clasificar_fs(r['fs'])})")

    graficar_comparacion(resultados)
