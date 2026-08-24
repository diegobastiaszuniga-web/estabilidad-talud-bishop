"""
Metodo de Bishop simplificado - Analisis de estabilidad de taludes
--------------------------------------------------------------------
Pipeline completo:
1. Definir el perfil del talud (superficie del terreno)
2. Definir el circulo de falla (centro, radio)
3. Encontrar automaticamente donde el circulo corta el terreno (entrada/salida)
4. Generar dovelas dentro de ese rango
5. Resolver FS de forma iterativa (Bishop simplificado)

Convencion de coordenadas: sistema cartesiano estandar (y hacia arriba = elevacion),
distinto del sistema de pantalla (y hacia abajo) usado en graficos SVG.
"""

import math
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 1. Perfil del talud
# ------------------------------------------------------------------

def interpolar_terreno(x, perfil):
    """
    Interpola linealmente la elevacion del terreno en una coordenada x.

    perfil: lista de tuplas (x, y) ordenadas de izquierda a derecha,
            definiendo el perfil del talud (ej: cresta, quiebre, pie).
    """
    xs = [p[0] for p in perfil]
    ys = [p[1] for p in perfil]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    return float(np.interp(x, xs, ys))


# ------------------------------------------------------------------
# 2. Circulo de falla
# ------------------------------------------------------------------

def elevacion_circulo(x, xc, yc, R):
    """
    Elevacion del arco de falla (parte inferior del circulo) en una coordenada x.
    Retorna None si x esta fuera del rango horizontal del circulo.
    """
    dentro_rango = R ** 2 - (x - xc) ** 2
    if dentro_rango < 0:
        return None
    return yc - math.sqrt(dentro_rango)


def encontrar_interseccion_circulo_terreno(xc, yc, R, perfil, lado="izquierdo"):
    """
    Encuentra el punto x donde el circulo de falla corta la superficie
    del terreno (punto de entrada o de salida), buscando la raiz de
    f(x) = elevacion_terreno(x) - elevacion_circulo(x).
    """
    xs_terreno = [p[0] for p in perfil]
    x_min, x_max = xs_terreno[0], xs_terreno[-1]

    def diferencia(x):
        y_circ = elevacion_circulo(x, xc, yc, R)
        if y_circ is None:
            return None
        return interpolar_terreno(x, perfil) - y_circ

    # Solo se evaluan puntos donde el circulo esta definido; los bordes
    # donde el circulo "desaparece" (fuera de su rango horizontal) se
    # descartan en vez de rellenarse con un valor centinela, que generaba
    # falsas raices justo en xc +/- R.
    xs_prueba = np.linspace(x_min, x_max, 400)
    puntos = [(x, diferencia(x)) for x in xs_prueba]
    puntos = [(x, v) for x, v in puntos if v is not None]

    raices = []
    for i in range(len(puntos) - 1):
        x_i, v_i = puntos[i]
        x_j, v_j = puntos[i + 1]
        if v_i == 0:
            raices.append(x_i)
        elif v_i * v_j < 0:
            raiz = brentq(diferencia, x_i, x_j)
            raices.append(raiz)

    if not raices:
        raise ValueError(
            "El circulo no interseca el perfil del terreno en el rango dado. "
            "Revisa xc, yc, R."
        )

    return min(raices) if lado == "izquierdo" else max(raices)


# ------------------------------------------------------------------
# 3. Generacion de dovelas
# ------------------------------------------------------------------

GAMMA_AGUA = 9.81  # kN/m3, peso unitario del agua


def elevacion_nivel_freatico(x, nivel_freatico):
    """
    Elevacion del nivel freatico en una coordenada x.

    nivel_freatico puede ser:
      - un numero (float/int): nivel freatico horizontal a esa elevacion
      - una lista de tuplas (x, y): una polilinea (igual que el perfil del
        talud), para representar un nivel freatico que sigue aprox. la
        topografia u otra forma arbitraria.
    """
    if isinstance(nivel_freatico, (int, float)):
        return float(nivel_freatico)
    return interpolar_terreno(x, nivel_freatico)


def generar_dovelas(xc, yc, R, perfil, n_dovelas=20, ru=0.0, nivel_freatico=None,
                     materiales=None, capas=None):
    """
    Genera n_dovelas dovelas verticales entre el punto de entrada y salida
    del circulo de falla sobre el perfil del terreno.

    Presion de poros -- dos formas de definirla (mutuamente excluyentes):
      - ru (por defecto): coeficiente simplificado, u = ru * gamma * h.
      - nivel_freatico: presion de poros hidrostatica real (ver mas abajo).

    Materiales -- dos formas de definirlos (mutuamente excluyentes):

    1) capas: lista de N capas ordenadas de la mas superficial a la mas
       profunda, cada una con su "piso" (limite inferior):
           capas = [
               {"piso": <numero o polilinea>, "c": .., "phi_deg": .., "gamma": ..},
               {"piso": <numero o polilinea>, "c": .., "phi_deg": .., "gamma": ..},
               {"piso": None, "c": .., "phi_deg": .., "gamma": ..},  # capa mas
                                                                       # profunda,
                                                                       # sin limite
           ]
       La ULTIMA capa debe tener "piso": None (se extiende hasta donde
       llegue el circulo). Cada "piso" es una linea geometrica (numero o
       polilinea, igual formato que nivel_freatico) -- estas fronteras NO
       son el nivel freatico, son contactos entre materiales.

    2) materiales: forma anterior, solo 2 capas (se mantiene por
       compatibilidad, internamente se convierte al formato de "capas"):
           materiales = {
               "contacto": <numero o polilinea>,
               "superior": {"c": .., "phi_deg": .., "gamma": ..},
               "inferior": {"c": .., "phi_deg": .., "gamma": ..},
           }

    Para cada dovela, el peso se calcula integrando el peso unitario de
    cada capa que la dovela atraviesa verticalmente. La resistencia al
    corte (c', phi') en la BASE de la dovela usa las propiedades de la
    capa en la que esa base realmente esta apoyada -- que es lo que
    importa para la resistencia, ya que la falla ocurre ahi.

    Si ni materiales ni capas se entregan (caso por defecto), el
    comportamiento es identico al de un unico material homogeneo, cuyo
    c/phi/gamma se entregan directamente al solver (calcular_fs_bishop).

    Retorna una lista de diccionarios con x_mid, b, h, alpha, ru, y_circulo,
    u_explicito, y (si corresponde) peso_por_ancho, c_dovela, phi_dovela_deg.
    """
    # Compatibilidad: convertir el formato antiguo de 2 capas al nuevo
    if materiales is not None and capas is None:
        capas = [
            {"piso": materiales["contacto"], **materiales["superior"]},
            {"piso": None, **materiales["inferior"]},
        ]

    x_izq = encontrar_interseccion_circulo_terreno(xc, yc, R, perfil, "izquierdo")
    x_der = encontrar_interseccion_circulo_terreno(xc, yc, R, perfil, "derecho")

    bordes = np.linspace(x_izq, x_der, n_dovelas + 1)
    dovelas = []

    for i in range(n_dovelas):
        x_ini, x_fin = bordes[i], bordes[i + 1]
        x_mid = 0.5 * (x_ini + x_fin)
        b = x_fin - x_ini

        y_terreno = interpolar_terreno(x_mid, perfil)
        y_circulo = elevacion_circulo(x_mid, xc, yc, R)
        h = y_terreno - y_circulo

        # alpha: angulo de la base de la dovela respecto a la horizontal.
        # sin(alpha) = (xc - x_mid) / R. El signo se define asi (centro
        # menos x, no al reves) porque en nuestra convencion el talud
        # desciende hacia +x: con esta definicion, alpha > 0 en las
        # dovelas del lado del pie (las que realmente arrastran la masa
        # hacia afuera), y Sum(W*sin(alpha)) resulta en el momento motor
        # neto correcto. Si tu perfil desciende hacia -x, invierte el signo.
        sin_alpha = max(-1.0, min(1.0, (xc - x_mid) / R))
        alpha = math.asin(sin_alpha)

        u_explicito = None
        if nivel_freatico is not None:
            y_nf = elevacion_nivel_freatico(x_mid, nivel_freatico)
            altura_agua = max(0.0, y_nf - y_circulo)
            u_explicito = GAMMA_AGUA * altura_agua

        peso_por_ancho = None
        c_dovela = None
        phi_dovela_deg = None
        if capas is not None:
            peso_por_ancho = 0.0
            y_techo_capa = y_terreno

            for capa in capas:
                if capa["piso"] is not None:
                    y_piso_capa = elevacion_nivel_freatico(x_mid, capa["piso"])
                else:
                    y_piso_capa = y_circulo - 1.0  # sin limite: llega hasta la base

                # Recortamos la capa al rango real de la dovela [y_circulo, y_techo_capa]
                y_piso_efectivo = max(y_piso_capa, y_circulo)
                if y_piso_efectivo < y_techo_capa:
                    h_en_capa = y_techo_capa - y_piso_efectivo
                    peso_por_ancho += capa["gamma"] * h_en_capa

                    # Si esta capa llega hasta la base de la dovela (o mas
                    # abajo), es la capa donde apoya la base -> define c, phi
                    if y_piso_capa <= y_circulo:
                        c_dovela = capa["c"]
                        phi_dovela_deg = capa["phi_deg"]

                y_techo_capa = y_piso_capa
                if y_piso_capa <= y_circulo:
                    break  # ya llegamos a la base, no hay mas capas relevantes

        dovelas.append({
            "x_mid": x_mid, "b": b, "h": h, "alpha": alpha, "ru": ru,
            "y_circulo": y_circulo,  # se guarda para poder ubicar el centro de
                                     # gravedad de la dovela (necesario en el
                                     # analisis pseudo-estatico sismico)
            "u_explicito": u_explicito,
            "peso_por_ancho": peso_por_ancho,
            "c_dovela": c_dovela,
            "phi_dovela_deg": phi_dovela_deg,
        })

    return dovelas



# ------------------------------------------------------------------
# 4. Solver iterativo de Bishop simplificado
# ------------------------------------------------------------------

def calcular_fs_bishop(dovelas, c, phi_deg, gamma, fs_inicial=1.0,
                        tol=1e-4, max_iter=100, relajacion=0.5, verbose=False):
    """
    Resuelve el Factor de Seguridad por el metodo de Bishop simplificado.
    FS aparece en ambos lados de la ecuacion (dentro de m_alpha), por lo
    que se resuelve por iteracion de punto fijo hasta convergencia.

    relajacion: factor de sub-relajacion (0 a 1). La iteracion directa
    (relajacion=1) puede oscilar sin converger quando m_alpha se acerca
    a cero para algunas dovelas; promediar el valor nuevo con el anterior
    (relajacion=0.5, tipico) estabiliza la convergencia.

    Nota sobre el signo: Sum(W*sin(alpha)) es el momento motor neto
    respecto al centro del circulo. Si dovelas fue generado con un
    circulo cuyo centro no corresponde a una direccion de deslizamiento
    fisicamente valida, este valor puede salir negativo o casi cero;
    eso indica que el circulo de prueba no es adecuado, no un error de
    calculo.

    Si las dovelas fueron generadas con el parametro 'materiales' (dos
    capas), c, phi_deg y gamma se ignoran para esas dovelas -- se usan
    en su lugar las propiedades por-dovela ('c_dovela', 'phi_dovela_deg',
    'peso_por_ancho') calculadas en generar_dovelas(). Los parametros
    c, phi_deg, gamma siguen siendo obligatorios porque se usan como
    valor por defecto para cualquier dovela sin esos campos (modo
    homogeneo, el caso normal).
    """
    fs = fs_inicial
    historial = [fs]

    for iteracion in range(max_iter):
        numerador = 0.0
        denominador = 0.0

        for d in dovelas:
            alpha = d["alpha"]
            b = d["b"]

            c_d = d["c_dovela"] if d.get("c_dovela") is not None else c
            phi_d_deg = d["phi_dovela_deg"] if d.get("phi_dovela_deg") is not None else phi_deg
            phi_d = math.radians(phi_d_deg)

            if d.get("peso_por_ancho") is not None:
                W = d["peso_por_ancho"] * b
            else:
                W = gamma * b * d["h"]

            m_alpha = math.cos(alpha) + (math.sin(alpha) * math.tan(phi_d)) / fs
            if abs(m_alpha) < 1e-6:
                m_alpha = 1e-6 if m_alpha >= 0 else -1e-6

            u = d["u_explicito"] if d.get("u_explicito") is not None else d["ru"] * gamma * d["h"]
            numerador += (c_d * b + (W - u * b) * math.tan(phi_d)) / m_alpha
            denominador += W * math.sin(alpha)

        if denominador <= 0:
            raise ValueError(
                "El momento motor neto (denominador) no es positivo: este "
                "circulo de prueba no representa una superficie de falla "
                "valida en la direccion esperada. Prueba otro xc/yc/R."
            )

        fs_bruto = numerador / denominador
        fs_nuevo = relajacion * fs_bruto + (1 - relajacion) * fs
        historial.append(fs_nuevo)

        if verbose:
            print(f"  Iteracion {iteracion + 1}: FS = {fs_nuevo:.5f}")

        if abs(fs_nuevo - fs) < tol:
            return fs_nuevo, historial

        fs = fs_nuevo

    print("Advertencia: no convergio dentro del numero maximo de iteraciones.")
    return fs, historial


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


def calcular_fs_bishop_pseudoestatico(dovelas, c, phi_deg, gamma, yc, R, kh=0.0,
                                       fs_inicial=1.0, tol=1e-4, max_iter=100,
                                       relajacion=0.5, verbose=False):
    """
    Igual que calcular_fs_bishop, pero agrega una fuerza horizontal
    pseudo-estatica kh*W sobre cada dovela (para simular un sismo).

    Esa fuerza horizontal tambien genera un momento respecto al centro
    del circulo O, pero su brazo de palanca es la distancia VERTICAL
    entre O y el centro de gravedad de la dovela (a diferencia del peso,
    cuyo brazo es la distancia horizontal, ya capturada en sin(alpha)).

    Termino sismico anadido al denominador (momento motor):
        kh * W * (yc - y_cg) / R
    donde y_cg = y_circulo + h/2 (centro de gravedad aproximado de la dovela).

    kh: coeficiente sismico horizontal (adimensional, fraccion de g).
        Valores tipicos usados en Chile para depositos de relaves (DS248)
        son del orden de 0.10-0.20 segun la zonificacion sismica del sitio.
    kh=0.0 reproduce exactamente el resultado de calcular_fs_bishop.

    Igual que en calcular_fs_bishop, si las dovelas incluyen propiedades
    por-dovela (modo 'materiales', dos capas), esas tienen prioridad sobre
    los c, phi_deg, gamma globales para esa dovela.
    """
    fs = fs_inicial
    historial = [fs]

    for iteracion in range(max_iter):
        numerador = 0.0
        denominador = 0.0

        for d in dovelas:
            alpha = d["alpha"]
            b = d["b"]

            c_d = d["c_dovela"] if d.get("c_dovela") is not None else c
            phi_d_deg = d["phi_dovela_deg"] if d.get("phi_dovela_deg") is not None else phi_deg
            phi_d = math.radians(phi_d_deg)

            if d.get("peso_por_ancho") is not None:
                W = d["peso_por_ancho"] * b
            else:
                W = gamma * b * d["h"]

            u = d["u_explicito"] if d.get("u_explicito") is not None else d["ru"] * gamma * d["h"]

            m_alpha = math.cos(alpha) + (math.sin(alpha) * math.tan(phi_d)) / fs
            if abs(m_alpha) < 1e-6:
                m_alpha = 1e-6 if m_alpha >= 0 else -1e-6

            numerador += (c_d * b + (W - u * b) * math.tan(phi_d)) / m_alpha

            y_cg = d["y_circulo"] + d["h"] / 2
            termino_gravedad = W * math.sin(alpha)
            termino_sismico = kh * W * (yc - y_cg) / R
            denominador += termino_gravedad + termino_sismico

        if denominador <= 0:
            raise ValueError(
                "El momento motor neto (con sismo incluido) no es positivo: "
                "este circulo de prueba no es valido para esta combinacion "
                "de parametros."
            )

        fs_bruto = numerador / denominador
        fs_nuevo = relajacion * fs_bruto + (1 - relajacion) * fs
        historial.append(fs_nuevo)

        if verbose:
            print(f"  Iteracion {iteracion + 1}: FS = {fs_nuevo:.5f}")

        if abs(fs_nuevo - fs) < tol:
            return fs_nuevo, historial

        fs = fs_nuevo

    print("Advertencia: no convergio dentro del numero maximo de iteraciones.")
    return fs, historial


# ------------------------------------------------------------------
# 5. Visualizacion
# ------------------------------------------------------------------

def graficar_talud_y_circulo(perfil, xc, yc, R, dovelas, fs):
    fig, ax = plt.subplots(figsize=(9, 6))

    xs_terreno = [p[0] for p in perfil]
    ys_terreno = [p[1] for p in perfil]
    ax.plot(xs_terreno, ys_terreno, color="black", linewidth=2, label="Superficie del talud")

    theta = np.linspace(0, 2 * np.pi, 300)
    x_circ = xc + R * np.cos(theta)
    y_circ = yc + R * np.sin(theta)
    mascara = y_circ <= max(ys_terreno) + 5
    ax.plot(x_circ[mascara], y_circ[mascara], "--", color="darkorange",
            linewidth=1.5, label="Superficie de falla circular")

    for d in dovelas:
        x = d["x_mid"]
        y_terreno = interpolar_terreno(x, perfil)
        y_circulo = elevacion_circulo(x, xc, yc, R)
        ax.plot([x, x], [y_circulo, y_terreno], color="gray", linewidth=0.6)

    ax.plot(xc, yc, "k+", markersize=10)
    ax.set_xlabel("Distancia horizontal (m)")
    ax.set_ylabel("Elevacion (m)")
    ax.set_title(f"Analisis de Bishop simplificado - FS = {fs:.3f}")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/bishop_talud_circulo.png", dpi=150)
    plt.show()


# ------------------------------------------------------------------
# Ejemplo de uso
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Perfil de un talud de deposito de relaves (puntos x,y en metros):
    # cresta plana, cara del talud, pie plano.
    perfil_talud = [
        (0, 30),
        (20, 30),
        (50, 0),
        (80, 0),
    ]

    # Circulo de falla de prueba (definido manualmente por ahora;
    # el siguiente paso de la ruta es buscar el circulo critico automaticamente)
    xc, yc, R = 30, 65, 40

    dovelas = generar_dovelas(xc, yc, R, perfil_talud, n_dovelas=20, ru=0.3)

    c = 5.0         # kPa, cohesion tipica de relaves
    phi_deg = 30.0  # grados, angulo de friccion tipico
    gamma = 18.0    # kN/m3, peso unitario tipico de relaves

    fs, historial = calcular_fs_bishop(dovelas, c, phi_deg, gamma, verbose=True)

    print(f"\nFactor de Seguridad (Bishop simplificado): {fs:.3f}")
    print(f"Convergio en {len(historial) - 1} iteraciones")

    graficar_talud_y_circulo(perfil_talud, xc, yc, R, dovelas, fs)
