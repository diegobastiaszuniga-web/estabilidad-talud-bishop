"""
API REST - Estabilidad de talud (Bishop simplificado)
--------------------------------------------------------------
Envuelve bishop_simplificado.py y busqueda_circulo_critico.py en
endpoints HTTP, para que un frontend (HTML/JS, otra app, o incluso
esta misma app de Streamlit) pueda pedir un calculo sin importar el
codigo Python directamente.

Ejecutar localmente con:
    uvicorn api:app --reload

Documentacion interactiva (Swagger) generada automaticamente en:
    http://localhost:8000/docs
"""

import math
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bishop_simplificado import (
    generar_dovelas,
    calcular_fs_bishop,
    calcular_fs_bishop_pseudoestatico,
    clasificar_fs,
)
from busqueda_circulo_critico import buscar_circulo_critico

app = FastAPI(
    title="API de Estabilidad de Talud",
    description="Calculo del Factor de Seguridad por el metodo de Bishop simplificado.",
    version="1.0.0",
)

# Permite que un frontend HTML/JS corriendo en otro origen (ej. localhost:5500,
# o un dominio distinto) pueda llamar a esta API sin ser bloqueado por CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Modelos de entrada/salida (Pydantic) -- definen el "contrato" JSON
# ------------------------------------------------------------------

class CirculoInput(BaseModel):
    xc: float = Field(..., description="Coordenada x del centro del circulo (m)")
    yc: float = Field(..., description="Coordenada y del centro del circulo (m)")
    R: float = Field(..., gt=0, description="Radio del circulo (m)")


class MaterialInput(BaseModel):
    c: float = Field(..., ge=0, description="Cohesion efectiva c' (kPa)")
    phi_deg: float = Field(..., ge=0, le=45, description="Angulo de friccion phi' (grados)")
    gamma: float = Field(18.0, gt=0, description="Peso unitario (kN/m3)")


class CalcularFSRequest(BaseModel):
    perfil: List[Tuple[float, float]] = Field(
        ..., min_length=2,
        description="Perfil del talud: lista de puntos (x,y) de izquierda a derecha"
    )
    circulo: CirculoInput
    material: MaterialInput
    ru: float = Field(0.0, ge=0, le=1, description="Coeficiente de presion de poros")
    kh: float = Field(0.0, ge=0, le=0.5, description="Coeficiente sismico horizontal")
    n_dovelas: int = Field(20, ge=6, le=100)


class DovelaOutput(BaseModel):
    x_mid: float
    h: float
    alpha_deg: float


class CalcularFSResponse(BaseModel):
    fs: float
    clasificacion: str
    dovelas: List[DovelaOutput]


class BuscarCriticoRequest(BaseModel):
    perfil: List[Tuple[float, float]] = Field(..., min_length=2)
    material: MaterialInput
    ru: float = Field(0.0, ge=0, le=1)
    kh: float = Field(0.0, ge=0, le=0.5)
    xc_min: float
    xc_max: float
    yc_min: float
    yc_max: float
    R_min: float
    R_max: float
    paso: float = Field(3.0, gt=0, description="Paso de la grilla de busqueda (m)")


class BuscarCriticoResponse(BaseModel):
    circulo: CirculoInput
    fs: float
    clasificacion: str


def _rango(minimo, maximo, paso):
    """Genera una lista de valores (busqueda_circulo_critico espera un iterable)."""
    valores = []
    v = minimo
    while v <= maximo:
        valores.append(v)
        v += paso
    return valores


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    """Chequeo simple de que la API esta corriendo."""
    return {"status": "ok"}


@app.post("/calcular-fs", response_model=CalcularFSResponse)
def calcular_fs(req: CalcularFSRequest):
    """
    Calcula el Factor de Seguridad para UN circulo de falla especifico
    (no busca el circulo critico -- para eso usa /buscar-circulo-critico).
    """
    try:
        dovelas = generar_dovelas(
            req.circulo.xc, req.circulo.yc, req.circulo.R, req.perfil,
            n_dovelas=req.n_dovelas, ru=req.ru,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if any(d["h"] <= 0 for d in dovelas):
        raise HTTPException(
            status_code=400,
            detail="El circulo no genera una masa deslizante valida sobre este perfil.",
        )

    try:
        if req.kh > 0:
            fs, _ = calcular_fs_bishop_pseudoestatico(
                dovelas, c=req.material.c, phi_deg=req.material.phi_deg,
                gamma=req.material.gamma, yc=req.circulo.yc, R=req.circulo.R, kh=req.kh,
            )
        else:
            fs, _ = calcular_fs_bishop(
                dovelas, c=req.material.c, phi_deg=req.material.phi_deg,
                gamma=req.material.gamma,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    dovelas_out = [
        DovelaOutput(x_mid=d["x_mid"], h=d["h"], alpha_deg=math.degrees(d["alpha"]))
        for d in dovelas
    ]

    return CalcularFSResponse(
        fs=round(fs, 4),
        clasificacion=clasificar_fs(fs),
        dovelas=dovelas_out,
    )


@app.post("/buscar-circulo-critico", response_model=BuscarCriticoResponse)
def buscar_circulo_critico_endpoint(req: BuscarCriticoRequest):
    """
    Busca el circulo de falla con el FS minimo dentro de los rangos
    (xc, yc, R) entregados. Puede tardar unos segundos segun el paso
    de la grilla y el tamano de los rangos.
    """
    paso = req.paso
    xc_rango = _rango(req.xc_min, req.xc_max, paso)
    yc_rango = _rango(req.yc_min, req.yc_max, paso)
    R_rango = _rango(req.R_min, req.R_max, paso)

    try:
        mejor = buscar_circulo_critico(
            req.perfil, c=req.material.c, phi_deg=req.material.phi_deg,
            gamma=req.material.gamma, ru=req.ru,
            xc_rango=xc_rango, yc_rango=yc_rango, R_rango=R_rango,
            kh=req.kh,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return BuscarCriticoResponse(
        circulo=CirculoInput(xc=mejor["xc"], yc=mejor["yc"], R=mejor["R"]),
        fs=round(mejor["fs"], 4),
        clasificacion=clasificar_fs(mejor["fs"]),
    )
