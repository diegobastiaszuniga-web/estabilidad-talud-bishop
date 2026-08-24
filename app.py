"""
App interactiva - Estabilidad de talud (Bishop simplificado)
----------------------------------------------------------------
Ejecutar localmente con:
    streamlit run app.py

Reutiliza directamente las funciones de bishop_simplificado.py y
busqueda_circulo_critico.py -- no se reescribe la logica de calculo,
solo se envuelve en una interfaz interactiva.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from bishop_simplificado import (
    generar_dovelas,
    calcular_fs_bishop,
    calcular_fs_bishop_pseudoestatico,
    clasificar_fs,
    interpolar_terreno,
    elevacion_circulo,
    elevacion_nivel_freatico,
)
from busqueda_circulo_critico import buscar_circulo_critico
from comparar_escenarios import escenarios as ESCENARIOS_PRESET, perfil_talud as PERFIL

# ------------------------------------------------------------------
# Configuracion general
# ------------------------------------------------------------------

st.set_page_config(page_title="Estabilidad de talud - Bishop", layout="wide")

GAMMA = 18.0  # kN/m3

COLOR_ESTABLE = "#0F6E56"
COLOR_MARGINAL = "#854F0B"
COLOR_INESTABLE = "#791F1F"
COLOR_AGUA = "#378ADD"


def color_por_fs(fs):
    if fs is None:
        return "gray"
    if fs >= 1.5:
        return COLOR_ESTABLE
    if fs >= 1.0:
        return COLOR_MARGINAL
    return COLOR_INESTABLE


def calcular_fs_circulo(xc, yc, R, ru, c, phi_deg, kh, nivel_freatico=None,
                         capas=None, n_dovelas=20):
    """
    Funcion central de calculo: genera las dovelas para un circulo dado y
    resuelve el FS (estatico o pseudo-estatico segun kh). Se reutiliza
    tanto en el panel principal como en el analisis de sensibilidad, para
    no duplicar esta logica en dos lugares.

    Retorna (fs, dovelas, error_msg). fs es None si hubo un error.
    """
    try:
        dovelas = generar_dovelas(xc, yc, R, PERFIL, n_dovelas=n_dovelas,
                                   ru=ru, nivel_freatico=nivel_freatico, capas=capas)
        if any(d["h"] <= 0 for d in dovelas):
            return None, None, "El circulo no genera una masa deslizante valida sobre este perfil."
        if kh > 0:
            fs, _ = calcular_fs_bishop_pseudoestatico(
                dovelas, c=c, phi_deg=phi_deg, gamma=GAMMA, yc=yc, R=R, kh=kh
            )
        else:
            fs, _ = calcular_fs_bishop(dovelas, c=c, phi_deg=phi_deg, gamma=GAMMA)
        return fs, dovelas, None
    except (ValueError, RuntimeError) as e:
        return None, None, str(e)


# ------------------------------------------------------------------
# Estado inicial (circulo de prueba por defecto)
# ------------------------------------------------------------------

if "circulo" not in st.session_state:
    st.session_state.circulo = {"xc": 72, "yc": 66, "R": 65}

# ------------------------------------------------------------------
# Barra lateral: parametros (comunes a ambas pestanas)
# ------------------------------------------------------------------


def aplicar_preset():
    """
    Callback del selector de escenario. Se ejecuta ANTES de que Streamlit
    vuelva a dibujar los sliders, por eso podemos escribir en session_state
    aqui y que los sliders (que usan esas mismas keys) tomen el valor nuevo.
    """
    seleccion = st.session_state.escenario_sel
    if seleccion in ESCENARIOS_PRESET:
        preset = ESCENARIOS_PRESET[seleccion]
        st.session_state.ru_slider = preset["ru"]
        st.session_state.c_slider = preset["c"]
        st.session_state.phi_slider = preset["phi_deg"]


st.sidebar.header("Escenario")
st.sidebar.selectbox(
    "Cargar condicion predefinida",
    ["Personalizado"] + list(ESCENARIOS_PRESET.keys()),
    key="escenario_sel",
    on_change=aplicar_preset,
    help="Elige un escenario para cargar sus valores de ru, c' y phi'. "
         "Luego puedes seguir ajustando los sliders manualmente.",
)
if st.session_state.escenario_sel in ESCENARIOS_PRESET:
    st.sidebar.caption(ESCENARIOS_PRESET[st.session_state.escenario_sel]["descripcion"])

st.sidebar.header("Presion de poros")
modo_nivel_freatico = st.sidebar.checkbox(
    "Usar nivel freatico explicito (en vez de ru)", value=False,
    help="Si esta activo, la presion de poros se calcula de forma hidrostatica "
         "real a partir de una profundidad de agua, en vez del coeficiente "
         "simplificado ru."
)

if modo_nivel_freatico:
    profundidad_nf = st.sidebar.slider(
        "Profundidad del nivel freatico bajo la superficie (m)", 0.0, 20.0, 5.0, 0.5,
        help="El nivel freatico se define paralelo a la superficie del talud, "
             "a esta profundidad constante bajo el terreno."
    )
    nivel_freatico = [(x, y - profundidad_nf) for x, y in PERFIL]
    ru = 0.0  # no se usa en este modo
else:
    ru = st.sidebar.slider("Coeficiente de presion de poros (ru)", 0.0, 0.6, 0.2, 0.05,
                            key="ru_slider",
                            help="0 = talud drenado. 0.3-0.5 = relave con nivel freatico alto.")
    nivel_freatico = None
    profundidad_nf = 5.0  # valor de referencia si se cambia de modo despues

st.sidebar.header("Parametros del material")
c = st.sidebar.slider("Cohesion c' (kPa)", 0.0, 15.0, 5.0, 0.5, key="c_slider")
phi_deg = st.sidebar.slider("Angulo de friccion phi' (grados)", 15.0, 40.0, 30.0, 0.5,
                             key="phi_slider")

st.sidebar.header("Materiales")
modo_multimaterial = st.sidebar.checkbox(
    "Usar multiples capas (relave / fundacion)", value=False,
    help="Si esta activo, se ignoran los sliders de c'/phi' de arriba y se "
         "usan las propiedades por capa definidas aqui abajo. Cada dovela "
         "usa el material de la capa en la que realmente apoya su base."
)

capas = None
if modo_multimaterial:
    n_capas = st.sidebar.selectbox("Numero de capas", [2, 3], index=0)

    propiedades_capas = []
    profundidades_contacto = []

    for i in range(n_capas):
        st.sidebar.markdown(f"**Capa {i + 1}**" + (" (mas profunda)" if i == n_capas - 1 else ""))
        col_c, col_phi, col_gamma = st.sidebar.columns(3)
        c_i = col_c.number_input(f"c'", value=5.0 + i * 7, step=0.5, key=f"c_capa_{i}", label_visibility="visible")
        phi_i = col_phi.number_input(f"phi'", value=30.0 + i * 2, step=0.5, key=f"phi_capa_{i}", label_visibility="visible")
        gamma_i = col_gamma.number_input(f"gamma", value=18.0 + i * 1, step=0.5, key=f"gamma_capa_{i}", label_visibility="visible")
        propiedades_capas.append({"c": c_i, "phi_deg": phi_i, "gamma": gamma_i})

        if i < n_capas - 1:
            profundidad_i = st.sidebar.slider(
                f"Profundidad del piso de la capa {i + 1} (m)", 0.0, 25.0,
                3.0 + i * 5.0, 0.5, key=f"prof_capa_{i}",
            )
            profundidades_contacto.append(profundidad_i)

    # Construir la lista de capas para generar_dovelas(): cada una con su
    # "piso" (linea de contacto bajo la superficie), salvo la ultima (None)
    capas = []
    for i, props in enumerate(propiedades_capas):
        if i < len(profundidades_contacto):
            piso = [(x, y - profundidades_contacto[i]) for x, y in PERFIL]
        else:
            piso = None
        capas.append({"piso": piso, **props})

    st.sidebar.caption(
        "Las profundidades deben ir aumentando de una capa a la siguiente "
        "(capa 1 mas somera, ultima capa sin limite inferior)."
    )

st.sidebar.header("Condicion sismica")
kh = st.sidebar.slider("Coeficiente sismico horizontal (kh)", 0.0, 0.30, 0.0, 0.01,
                        help="0 = condicion estatica. Valores tipicos en Chile para "
                             "depositos de relaves (DS248): 0.10-0.20 segun zonificacion.")

st.sidebar.header("Circulo de falla")
modo_manual = st.sidebar.checkbox("Definir circulo manualmente", value=False)

if modo_manual:
    xc = st.sidebar.slider("Centro xc (m)", 10, 110, st.session_state.circulo["xc"])
    yc = st.sidebar.slider("Centro yc (m)", 40, 150, st.session_state.circulo["yc"])
    R = st.sidebar.slider("Radio R (m)", 30, 140, st.session_state.circulo["R"])
    st.session_state.circulo = {"xc": xc, "yc": yc, "R": R}
else:
    etiqueta_boton = ("Buscar circulo critico (sismico)" if kh > 0
                       else "Buscar circulo critico (estatico)")
    if st.sidebar.button(etiqueta_boton):
        with st.spinner("Buscando el circulo con FS minimo..."):
            try:
                # Nota: la busqueda de circulo critico (busqueda_circulo_critico.py)
                # todavia trabaja en modo ru -- si estas en modo nivel freatico
                # explicito, se usa un ru aproximado solo para esta busqueda de
                # geometria; el FS final que ves si se calcula con el nivel
                # freatico real.
                ru_para_busqueda = ru if not modo_nivel_freatico else 0.3
                mejor = buscar_circulo_critico(
                    PERFIL, c=c, phi_deg=phi_deg, gamma=GAMMA, ru=ru_para_busqueda,
                    xc_rango=range(15, 90, 3),
                    yc_rango=range(45, 120, 3),
                    R_rango=range(35, 120, 3),
                    kh=kh,
                )
                st.session_state.circulo = {"xc": mejor["xc"], "yc": mejor["yc"], "R": mejor["R"]}
            except RuntimeError as e:
                st.sidebar.error(str(e))
    st.sidebar.caption(
        "La busqueda usa el kh actual: si subes el sismo y vuelves a buscar, "
        "el circulo critico puede cambiar de posicion (no siempre lo hace)."
    )

xc = st.session_state.circulo["xc"]
yc = st.session_state.circulo["yc"]
R = st.session_state.circulo["R"]
st.sidebar.caption(f"Circulo actual: xc={xc}, yc={yc}, R={R}")

# ------------------------------------------------------------------
# Calculo del FS con los parametros actuales
# ------------------------------------------------------------------

fs, dovelas, error_msg = calcular_fs_circulo(xc, yc, R, ru, c, phi_deg, kh, nivel_freatico, capas)

# ------------------------------------------------------------------
# Layout principal: dos pestanas
# ------------------------------------------------------------------

st.title("Estabilidad de talud — metodo de Bishop simplificado")

tab_puntual, tab_sensibilidad, tab_validacion = st.tabs(
    ["Analisis puntual", "Analisis de sensibilidad", "Validacion contra benchmark"]
)

with tab_puntual:
    col_resultado, col_grafico = st.columns([1, 2])

    with col_resultado:
        st.subheader("Resultado")
        if fs is not None:
            color = color_por_fs(fs)
            condicion = f"Sismica (kh={kh:.2f})" if kh > 0 else "Estatica"
            st.caption(f"Condicion evaluada: {condicion}")
            st.markdown(f"<h1 style='color:{color}'>FS = {fs:.3f}</h1>", unsafe_allow_html=True)
            st.markdown(f"**{clasificar_fs(fs)}**")
            st.progress(min(fs / 2.0, 1.0))
            if kh > 0:
                st.caption("Referencia habitual: FS >= 1.0-1.1 en condicion sismica "
                           "(el umbral es menor que en condicion estatica).")
        else:
            st.error(f"No se pudo calcular el FS: {error_msg}")

        st.divider()
        st.caption("Referencias: FS < 1.0 inestable · FS 1.0-1.5 marginal · FS >= 1.5 estable")

    with col_grafico:
        fig_plotly = go.Figure()

        xs_terreno = [p[0] for p in PERFIL]
        ys_terreno = [p[1] for p in PERFIL]
        fig_plotly.add_trace(go.Scatter(
            x=xs_terreno, y=ys_terreno, mode="lines", name="Superficie del talud",
            line=dict(color="#ECEBE6", width=3),
            hovertemplate="x=%{x:.1f} m<br>elevacion=%{y:.1f} m<extra></extra>",
        ))

        if modo_nivel_freatico:
            xs_nf = [p[0] for p in nivel_freatico]
            ys_nf = [p[1] for p in nivel_freatico]
            fig_plotly.add_trace(go.Scatter(
                x=xs_nf, y=ys_nf, mode="lines", name="Nivel freatico",
                line=dict(color=COLOR_AGUA, width=2, dash="solid"),
                hovertemplate="Nivel freatico<br>x=%{x:.1f} m<br>elevacion=%{y:.1f} m<extra></extra>",
            ))

        if modo_multimaterial and capas is not None:
            # Un color distinto por cada linea de contacto (se repite si hay
            # mas contactos que colores en la lista, aunque hoy el maximo
            # son 2 contactos con 3 capas). Cambia estos hex para ajustar
            # la paleta.
            COLORES_CONTACTO = ["#9B8FE8", "#E07BA8", "#5DCAA5"]
            for i, capa_i in enumerate(capas):
                if capa_i["piso"] is not None:
                    color_contacto = COLORES_CONTACTO[i % len(COLORES_CONTACTO)]
                    xs_c = [p[0] for p in capa_i["piso"]]
                    ys_c = [p[1] for p in capa_i["piso"]]
                    fig_plotly.add_trace(go.Scatter(
                        x=xs_c, y=ys_c, mode="lines", name=f"Contacto capa {i+1}/{i+2}",
                        line=dict(color=color_contacto, width=1.5, dash="dot"),
                        hovertemplate=f"Contacto capa {i+1}/{i+2}<br>x=%{{x:.1f}} m<br>elevacion=%{{y:.1f}} m<extra></extra>",
                    ))

        theta = np.linspace(0, 2 * np.pi, 300)
        x_circ = xc + R * np.cos(theta)
        y_circ = yc + R * np.sin(theta)
        mascara = y_circ <= max(ys_terreno) + 10
        color_circulo = color_por_fs(fs)
        fig_plotly.add_trace(go.Scatter(
            x=x_circ[mascara], y=y_circ[mascara], mode="lines", name="Superficie de falla",
            line=dict(color=color_circulo, width=2, dash="dash"),
            hovertemplate="Circulo de falla<br>x=%{x:.1f} m<br>elevacion=%{y:.1f} m<extra></extra>",
        ))

        if dovelas is not None and fs is not None:
            for i, d in enumerate(dovelas):
                x = d["x_mid"]
                y_terreno_d = interpolar_terreno(x, PERFIL)
                y_circulo_d = elevacion_circulo(x, xc, yc, R)
                c_info = d["c_dovela"] if d.get("c_dovela") is not None else c
                phi_info = d["phi_dovela_deg"] if d.get("phi_dovela_deg") is not None else phi_deg
                texto_hover = (
                    f"Dovela {i+1}<br>"
                    f"x = {x:.1f} m<br>"
                    f"h = {d['h']:.2f} m<br>"
                    f"alpha = {np.degrees(d['alpha']):.1f} grados<br>"
                    f"c' = {c_info:.1f} kPa<br>"
                    f"phi' = {phi_info:.1f} grados"
                )
                fig_plotly.add_trace(go.Scatter(
                    x=[x, x], y=[y_circulo_d, y_terreno_d], mode="lines",
                    line=dict(color="gray", width=1),
                    showlegend=False, hovertemplate=texto_hover + "<extra></extra>",
                ))

        fig_plotly.add_trace(go.Scatter(
            x=[xc], y=[yc], mode="markers", name="Centro del circulo",
            marker=dict(symbol="cross", size=10, color="black"),
            hovertemplate=f"Centro O<br>xc={xc}, yc={yc}, R={R}<extra></extra>",
        ))

        fig_plotly.update_layout(
            title=f"FS = {fs:.2f}" if fs is not None else "Circulo invalido para este perfil",
            xaxis_title="Distancia horizontal (m)",
            yaxis_title="Elevacion (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),  # mismo aspecto que set_aspect('equal')
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=520,
            margin=dict(l=10, r=10, t=60, b=10),
        )

        st.plotly_chart(fig_plotly, use_container_width=True)

    st.divider()
    st.caption(
        "Modelo de talud finito con superficie de falla circular (Bishop simplificado). "
        "El circulo critico se busca por grilla, con filtros de ancho minimo y alpha maximo "
        "para descartar superficies no representativas."
    )

with tab_sensibilidad:
    st.subheader("¿Que tan sensible es el FS a un parametro?")
    st.caption(
        "Se mantiene fijo el circulo actual y todos los demas parametros en su "
        "valor de la barra lateral, y se varia solo el parametro elegido."
    )

    opciones_parametro = {
        "Cohesion c' (kPa)": "c",
        "Angulo de friccion phi' (grados)": "phi_deg",
        "Coeficiente sismico kh": "kh",
    }
    if not modo_nivel_freatico:
        opciones_parametro["Coeficiente de presion de poros (ru)"] = "ru"
    else:
        opciones_parametro["Profundidad del nivel freatico (m)"] = "profundidad_nf"

    etiqueta_parametro = st.selectbox("Parametro a variar", list(opciones_parametro.keys()))
    parametro = opciones_parametro[etiqueta_parametro]

    rangos_por_defecto = {
        "c": (0.0, 15.0, c),
        "phi_deg": (15.0, 40.0, phi_deg),
        "kh": (0.0, 0.30, kh),
        "ru": (0.0, 0.6, ru),
        "profundidad_nf": (0.0, 20.0, profundidad_nf),
    }
    val_min, val_max, val_actual = rangos_por_defecto[parametro]

    col_a, col_b = st.columns(2)
    with col_a:
        rango_min = st.number_input("Desde", value=float(val_min), step=0.5)
    with col_b:
        rango_max = st.number_input("Hasta", value=float(val_max), step=0.5)

    if st.button("Calcular curva de sensibilidad"):
        valores = np.linspace(rango_min, rango_max, 25)
        fs_valores = []

        with st.spinner("Calculando FS para cada valor..."):
            for v in valores:
                kwargs = dict(ru=ru, c=c, phi_deg=phi_deg, kh=kh, nivel_freatico=nivel_freatico)
                if parametro == "profundidad_nf":
                    kwargs["nivel_freatico"] = [(x, y - v) for x, y in PERFIL]
                else:
                    kwargs[parametro] = v

                fs_v, _, _ = calcular_fs_circulo(xc, yc, R, **kwargs)
                fs_valores.append(fs_v)

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        valores_validos = [v for v, f in zip(valores, fs_valores) if f is not None]
        fs_validos = [f for f in fs_valores if f is not None]

        ax2.plot(valores_validos, fs_validos, "-o", color="#534AB7", markersize=4)
        ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="FS = 1.0")
        ax2.axhline(y=1.5, color="gray", linestyle=":", linewidth=1, label="FS = 1.5")
        ax2.axvline(x=val_actual, color=COLOR_MARGINAL, linestyle="-", linewidth=1,
                    alpha=0.6, label=f"Valor actual ({val_actual:.2f})")

        ax2.set_xlabel(etiqueta_parametro)
        ax2.set_ylabel("Factor de Seguridad (FS)")
        ax2.set_title(f"Sensibilidad del FS a {etiqueta_parametro}")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig2)

        if len(fs_valores) != len(fs_validos):
            st.caption(
                f"{len(fs_valores) - len(fs_validos)} de {len(fs_valores)} puntos no se "
                "pudieron calcular (circulo invalido para ese valor) y se omitieron."
            )

with tab_validacion:
    st.subheader("Validacion contra un caso publicado (ACADS Simple Slope)")
    st.markdown(
        "Este es un caso de verificacion estandar de la industria, publicado por "
        "**Giam y Donald (1989)** para la Association for Computer Aided Design "
        "(ACADS), y reproducido en el manual de verificacion oficial de "
        "**GeoStudio SLOPE/W**. Es un talud homogeneo, sin agua, con propiedades "
        "conocidas -- util para comprobar que este codigo da resultados "
        "razonables frente a un software comercial."
    )

    PERFIL_ACADS = [(20, 35), (40, 35), (60, 25), (70, 25)]
    C_ACADS, PHI_ACADS, GAMMA_ACADS = 3.0, 19.6, 20.0
    FS_ACADS_PUBLICADO = 1.00
    FS_SLOPEW_BISHOP = 0.963

    col_val_a, col_val_b = st.columns([1, 1])

    with col_val_a:
        st.markdown("**Geometria y material (segun el benchmark)**")
        st.markdown(
            "- Perfil (espejado para calzar con la convencion de signo del codigo): "
            f"`{PERFIL_ACADS}`\n"
            f"- Cohesion c' = {C_ACADS} kPa\n"
            f"- Angulo de friccion phi' = {PHI_ACADS} grados\n"
            f"- Peso unitario gamma = {GAMMA_ACADS} kN/m3\n"
            "- Sin presion de poros, sin sismo"
        )

        if st.button("Correr validacion"):
            with st.spinner("Buscando el circulo critico para el caso ACADS..."):
                mejor_val = buscar_circulo_critico(
                    PERFIL_ACADS, c=C_ACADS, phi_deg=PHI_ACADS, gamma=GAMMA_ACADS, ru=0.0,
                    xc_rango=range(50, 70, 1),
                    yc_rango=range(42, 62, 1),
                    R_rango=range(15, 35, 1),
                    n_dovelas=25,
                )
                st.session_state.resultado_validacion = mejor_val

    with col_val_b:
        if "resultado_validacion" in st.session_state:
            r = st.session_state.resultado_validacion
            fs_propio = r["fs"]
            diff_acads = 100 * (fs_propio - FS_ACADS_PUBLICADO) / FS_ACADS_PUBLICADO
            diff_slopew = 100 * (fs_propio - FS_SLOPEW_BISHOP) / FS_SLOPEW_BISHOP

            st.markdown("**Resultado**")
            st.markdown(f"Circulo critico: xc={r['xc']}, yc={r['yc']}, R={r['R']}")

            tabla_md = (
                "| Fuente | FS | Diferencia |\n"
                "|---|---|---|\n"
                f"| ACADS (1989) | {FS_ACADS_PUBLICADO:.3f} | {diff_acads:+.1f}% |\n"
                f"| SLOPE/W (Bishop) | {FS_SLOPEW_BISHOP:.3f} | {diff_slopew:+.1f}% |\n"
                f"| **Este codigo** | **{fs_propio:.3f}** | -- |\n"
            )
            st.markdown(tabla_md)
        else:
            st.info("Aprieta 'Correr validacion' para calcular y comparar.")

    if "resultado_validacion" in st.session_state:
        r = st.session_state.resultado_validacion
        fig3, ax3 = plt.subplots(figsize=(7, 5))

        xs_v = [p[0] for p in PERFIL_ACADS]
        ys_v = [p[1] for p in PERFIL_ACADS]
        ax3.plot(xs_v, ys_v, color="black", linewidth=2, label="Superficie del talud (ACADS)")

        theta = np.linspace(0, 2 * np.pi, 300)
        x_circ = r["xc"] + r["R"] * np.cos(theta)
        y_circ = r["yc"] + r["R"] * np.sin(theta)
        mascara = y_circ <= max(ys_v) + 10
        ax3.plot(x_circ[mascara], y_circ[mascara], "--", color=COLOR_MARGINAL, linewidth=2,
                 label="Circulo critico encontrado")

        ax3.set_xlabel("Distancia horizontal (m)")
        ax3.set_ylabel("Elevacion (m)")
        ax3.set_title(f"Validacion ACADS Simple Slope - FS = {r['fs']:.3f}")
        ax3.legend(fontsize=9)
        ax3.set_aspect("equal", adjustable="box")
        ax3.grid(True, alpha=0.3)

        st.pyplot(fig3)

    st.caption(
        "Fuente: Giam, P.S.K. y Donald, I.B. (1989), 'Example problems for testing "
        "soil slope stability programs', Monash University. Reproducido en el "
        "manual de verificacion de GeoStudio SLOPE/W (Seequent/Bentley, 2022)."
    )
