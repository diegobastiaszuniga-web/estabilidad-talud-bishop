"""
Modelo de talud infinito - Factor de Seguridad (FS)
----------------------------------------------------
Paso 1 y 3 de la ruta: modelo mas simple de estabilidad de taludes,
util como punto de partida antes de pasar a Bishop simplificado.

Supuestos del modelo de talud infinito:
- El talud es "infinito" (largo >> profundidad de la superficie de falla)
- La superficie de falla es paralela a la superficie del talud
- El analisis se hace sobre una columna de suelo de ancho unitario

Formula general (con presion de poros via nivel freatico o ru):

    FS = [c' + (gamma * z * cos(beta)^2 - u) * tan(phi')] / (gamma * z * sin(beta) * cos(beta))

donde:
    c'      : cohesion efectiva del material (kPa)
    phi'    : angulo de friccion interna efectivo (grados)
    gamma   : peso unitario del material (kN/m3)
    z       : profundidad vertical de la superficie de falla (m)
    beta    : angulo del talud respecto a la horizontal (grados)
    u       : presion de poros en la superficie de falla (kPa)
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def calcular_fs_talud_infinito(c, phi_deg, gamma, z, beta_deg, ru=0.0):
    """
    Calcula el Factor de Seguridad (FS) para un talud infinito.

    Parametros
    ----------
    c : float
        Cohesion efectiva (kPa)
    phi_deg : float
        Angulo de friccion interna efectivo (grados)
    gamma : float
        Peso unitario del material (kN/m3)
    z : float
        Profundidad vertical de la superficie de falla (m)
    beta_deg : float
        Angulo del talud (grados)
    ru : float, opcional
        Coeficiente de presion de poros (0 = seco, tipico 0.3-0.5 en relaves saturados)
        ru = u / (gamma * z)

    Retorna
    -------
    fs : float
        Factor de Seguridad. FS >= 1.5 se considera aceptable en la mayoria
        de las normativas para condicion estatica; FS >= 1.0-1.1 para
        condicion pseudo-estatica sismica (revisar normativa aplicable, ej. DS248).
    """
    beta = math.radians(beta_deg)
    phi = math.radians(phi_deg)

    # Esfuerzo normal total sobre el plano de falla
    sigma_n = gamma * z * math.cos(beta) ** 2

    # Presion de poros a partir de ru
    u = ru * gamma * z

    # Esfuerzo normal efectivo
    sigma_n_efectivo = sigma_n - u

    # Esfuerzo de corte actuante (motor de la falla)
    tau_actuante = gamma * z * math.sin(beta) * math.cos(beta)

    # Resistencia al corte disponible (Mohr-Coulomb)
    tau_resistente = c + sigma_n_efectivo * math.tan(phi)

    fs = tau_resistente / tau_actuante
    return fs


def clasificar_fs(fs):
    """Devuelve una interpretacion cualitativa rapida del FS."""
    if fs < 1.0:
        return "Inestable (falla teorica)"
    elif fs < 1.3:
        return "Marginal / requiere revision"
    elif fs < 1.5:
        return "Aceptable con reservas"
    else:
        return "Estable"


def graficar_sensibilidad_beta(c, phi_deg, gamma, z, ru=0.0, beta_min=15, beta_max=45):
    """
    Grafica como varia el FS al cambiar el angulo del talud (beta),
    manteniendo fijos los demas parametros.
    """
    betas = np.linspace(beta_min, beta_max, 100)
    fs_valores = [calcular_fs_talud_infinito(c, phi_deg, gamma, z, b, ru) for b in betas]

    plt.figure(figsize=(7, 5))
    plt.plot(betas, fs_valores, linewidth=2)
    plt.axhline(y=1.0, linestyle="--", label="FS = 1.0 (falla teorica)")
    plt.axhline(y=1.5, linestyle="--", label="FS = 1.5 (referencia estatica)")
    plt.xlabel("Angulo del talud, beta (grados)")
    plt.ylabel("Factor de Seguridad (FS)")
    plt.title("Sensibilidad del FS al angulo del talud - Modelo talud infinito")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/sensibilidad_beta_fs.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Ejemplo: talud de un deposito de relaves, condicion saturada
    # ------------------------------------------------------------
    c = 5.0          # kPa, cohesion baja tipica de relaves
    phi_deg = 30.0    # grados, angulo de friccion tipico de relaves finos/arenosos
    gamma = 18.0      # kN/m3, peso unitario tipico de relaves
    z = 4.0           # m, profundidad de la superficie de falla
    beta_deg = 25.0   # grados, angulo del talud
    ru = 0.3          # coeficiente de presion de poros (relave con nivel freatico alto)

    fs = calcular_fs_talud_infinito(c, phi_deg, gamma, z, beta_deg, ru)

    print(f"Factor de Seguridad (FS): {fs:.3f}")
    print(f"Interpretacion: {clasificar_fs(fs)}")

    # Sensibilidad: como cambia el FS si el talud es mas o menos inclinado
    graficar_sensibilidad_beta(c, phi_deg, gamma, z, ru)
