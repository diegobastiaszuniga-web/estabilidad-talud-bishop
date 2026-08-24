# Estabilidad de Talud — Método de Bishop Simplificado

Herramienta de análisis de estabilidad de taludes (mina/relaves) construida desde cero en Python, implementando el método de dovelas de Bishop simplificado con búsqueda de círculo crítico, condición sísmica pseudo-estática, nivel freático explícito, y soporte para dos materiales (ej. relave sobre fundación natural).

**[Ver la app en vivo →](https://estabilidad-talud-bishop-33jwofmitv6tbc9i2smmxf.streamlit.app)**

## Qué hace

- Cálculo del Factor de Seguridad (FS) por el método de Bishop simplificado, con solver iterativo propio (sin depender de librerías de estabilidad de taludes).
- Búsqueda automática del círculo de falla crítico (grid search), con filtros geométricos para descartar superficies no representativas.
- Condición sísmica pseudo-estática (coeficiente `kh`), siguiendo la práctica habitual en normativas de depósitos de relaves (ej. DS248 en Chile).
- Presión de poros: por coeficiente `ru` simplificado, o por nivel freático explícito con cálculo hidrostático real.
- Soporte para **N materiales/estratos** (ej. relave suelto → relave consolidado → fundación resistente), con propiedades por dovela según el material en que apoya su base.
- Análisis de sensibilidad interactivo (FS vs. cualquier parámetro).
- **Suite de tests automatizados** (`pytest`) con 9 tests: regresión, casos límite, y las dos validaciones contra literatura publicada corriendo automáticamente.
- Validado contra dos casos publicados de forma independiente:

| Benchmark | Fuente | FS/kc publicado | Este código | Diferencia |
|---|---|---|---|---|
| ACADS Simple Slope | Giam & Donald (1989) / GeoStudio SLOPE/W | FS = 1.00 | FS = 0.985 | -1.5% |
| Coeficiente sísmico crítico | Loukidis, Bandini & Salgado (2005) | kc = 0.114 | FS(kc) = 0.9998 | -0.02% |

## Stack técnico

- **Cálculo**: Python puro (numpy, scipy para búsqueda de raíces)
- **Interfaz interactiva**: Streamlit + Plotly
- **API REST**: FastAPI (expone el motor de cálculo vía HTTP)
- **Frontend alternativo**: HTML/JS puro consumiendo la API

## Estructura del proyecto

```
bishop_simplificado.py       # Motor de calculo: geometria, dovelas, solvers Bishop
busqueda_circulo_critico.py  # Busqueda en grilla del circulo critico
comparar_escenarios.py       # Comparacion de escenarios (favorable/intermedio/desfavorable)
talud_infinito.py            # Modelo simplificado de talud infinito (punto de partida)
validacion_acads.py          # Validacion contra benchmark ACADS
validacion_loukidis.py       # Validacion contra benchmark sismico Loukidis et al.
test_bishop.py                # Suite de tests automatizados (pytest)
app.py                       # App interactiva (Streamlit)
api.py                       # API REST (FastAPI)
frontend.html                # Frontend alternativo en HTML/JS puro
```

## Correr localmente

```bash
pip install -r requirements.txt

# App interactiva
streamlit run app.py

# API REST (en otra terminal)
uvicorn api:app --reload

# Suite de tests
pytest test_bishop.py -v
```

## Limitaciones (léelo antes de usarlo para algo real)

Este es un proyecto educativo/de portafolio, no un reemplazo de software comercial validado (GeoStudio, Slide2, etc.) para entregables de ingeniería con firma profesional. En particular:

- Solo superficies de falla **circulares** (no Morgenstern-Price ni superficies arbitrarias).
- Sin acople con modelo de flujo de agua subterránea (el nivel freático se define, no se calcula).
- La búsqueda de círculo crítico usa grid search por fuerza bruta, no optimización avanzada.

## Autor

Diego Bastías Zúñiga — Geólogo (Universidad Mayor, 2026)
