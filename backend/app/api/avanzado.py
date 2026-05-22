"""
API Router — IRG (Índice de Riesgo Global) + IA Multi-Modelo
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from app.services.irg_service import calcular_irg
from app.services.ia_service import analizar_con_ia, ModeloIA, MODELOS_INFO
from app.services.riesgo_service import calcular_riesgo_zona, _simular_datos_meteorologicos
from app.services.openmeteo_service import obtener_meteo_real, COORDS_ZONAS
from app.services.consultas_service import responder_consulta
from app.services.database import guardar_consulta

router_irg = APIRouter()
router_ia  = APIRouter()


# ── IRG ──────────────────────────────────────────────────────────────────────

@router_irg.get("/zona/{zona_id}")
async def get_irg(zona_id: str, hora: Optional[int] = None):
    """Calcula el Índice de Riesgo Global para una zona"""
    try:
        meteo = await obtener_meteo_real(zona_id)
    except Exception:
        meteo = _simular_datos_meteorologicos(zona_id)

    resultado = calcular_irg(meteo, {}, hora=hora, zona_id=zona_id)

    return {
        "zona_id": zona_id,
        "irg": resultado.irg,
        "nivel": resultado.nivel,
        "top_factores": resultado.top_factores,
        "alertas_irg": resultado.alertas_irg,
        "contexto_local": resultado.contexto_local,
        "timestamp": resultado.timestamp.isoformat(),
        "variables": {
            k: {
                "nombre": v.nombre, "valor": v.valor,
                "valor_raw": round(v.valor_raw, 2),
                "unidad": v.unidad, "peso": v.peso,
                "contribucion": round(v.contribucion, 5),
                "icono": v.icono, "descripcion": v.descripcion,
            }
            for k, v in resultado.variables.items()
        },
    }


@router_irg.get("/dashboard/{zona_id}")
async def get_irg_dashboard(zona_id: str):
    """IRG compacto para widget del dashboard"""
    try:
        meteo = await obtener_meteo_real(zona_id)
    except Exception:
        meteo = _simular_datos_meteorologicos(zona_id)

    r = calcular_irg(meteo, {}, zona_id=zona_id)
    return {
        "irg": r.irg,
        "irg_pct": round(r.irg * 100, 1),
        "nivel": r.nivel,
        "top_3": r.top_factores[:3],
        "n_alertas": len(r.alertas_irg),
        "turismo": r.contexto_local.get("turismo_nivel"),
        "hora": r.contexto_local.get("hora"),
    }


# ── IA Multi-Modelo ────────────────────────────────────────────────────────

class SolicitudIA(BaseModel):
    zona_id: str = "guatape"
    modelo: str = "local"
    api_key: Optional[str] = None
    lat: float = 6.2336
    lon: float = -75.1567


@router_ia.get("/modelos")
async def get_modelos():
    """Lista todos los modelos IA disponibles"""
    return {
        "modelos": [
            {
                "id": modelo.value,
                **info,
            }
            for modelo, info in MODELOS_INFO.items()
        ]
    }


@router_ia.post("/analizar")
async def analizar(solicitud: SolicitudIA):
    """
    Ejecuta análisis IA sobre el estado actual de riesgo.
    Combina datos internos (sensores, IRG, predicciones ML) 
    con datos externos (Open-Meteo gratuito).
    """
    # Validar modelo
    try:
        modelo_enum = ModeloIA(solicitud.modelo)
    except ValueError:
        raise HTTPException(400, f"Modelo '{solicitud.modelo}' no válido. Use: {[m.value for m in ModeloIA]}")

    # Recolectar datos internos
    zona = solicitud.zona_id
    try:
        meteo = await obtener_meteo_real(zona, solicitud.lat, solicitud.lon)
    except Exception:
        meteo = _simular_datos_meteorologicos(zona)

    irg_resultado = calcular_irg(meteo, {}, zona_id=zona)
    preds_resultado = await calcular_riesgo_zona(zona)

    # Construir payload combinado para el modelo IA
    datos_combinados = {
        "zona": zona,
        "sensores": {
            "lluvia24":  round(meteo.get("lluvia_24h_mm", 0), 1),
            "embalse":   round(meteo.get("nivel_embalse_pct", 0), 1),
            "temp":      round(meteo.get("temperatura_c", 0), 1),
            "hum":       round(meteo.get("humedad_suelo", 0) * 100, 1),
            "viento":    round(meteo.get("velocidad_viento_ms", 0), 1),
        },
        "irg": {
            "irg": irg_resultado.irg,
            "nivel": irg_resultado.nivel,
            "top_factores": irg_resultado.top_factores,
        },
        "alertas_irg": irg_resultado.alertas_irg,
        "predicciones": preds_resultado.get("predicciones", []),
        "contexto_local": irg_resultado.contexto_local,
        "factor_enso": 1.04,
        "timestamp": datetime.utcnow().isoformat(),
    }

    resultado = await analizar_con_ia(
        datos_combinados,
        modelo=modelo_enum,
        api_key=solicitud.api_key,
        lat=solicitud.lat,
        lon=solicitud.lon,
    )

    return {
        "zona_id": zona,
        "modelo": solicitud.modelo,
        "analisis": resultado,
        "datos_entrada": {
            "irg_pct": round(irg_resultado.irg * 100, 1),
            "nivel_irg": irg_resultado.nivel,
            "n_alertas": len(irg_resultado.alertas_irg),
            "datos_externos": resultado.get("datos_externos_disponibles", False),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@router_ia.get("/analizar-rapido/{zona_id}")
async def analizar_rapido(zona_id: str, modelo: str = "local"):
    """Análisis rápido GET (sin API key — siempre usa motor local o simulado)"""
    solicitud = SolicitudIA(zona_id=zona_id, modelo=modelo)
    return await analizar(solicitud)


# ── Consultas en lenguaje natural ─────────────────────────────────────────────

class SolicitudConsulta(BaseModel):
    pregunta: str
    zona_id: str = "guatape"
    lat: Optional[float] = None
    lon: Optional[float] = None


@router_ia.post("/consultar")
async def consultar(solicitud: SolicitudConsulta):
    """
    Responde preguntas en lenguaje natural sobre clima y riesgo.
    Ejemplos:
      - '¿Cuándo va a escampar?'
      - '¿Va a llover hoy?'
      - '¿Qué temperatura hace?'
      - '¿Hay riesgo de deslizamiento?'
    """
    if not solicitud.pregunta.strip():
        raise HTTPException(400, "La pregunta no puede estar vacía")

    coords = COORDS_ZONAS.get(solicitud.zona_id, (6.2336, -75.1567))
    lat = solicitud.lat or coords[0]
    lon = solicitud.lon or coords[1]

    respuesta = await responder_consulta(
        solicitud.pregunta, solicitud.zona_id, lat, lon
    )

    # Guardar en BD para histórico (fire-and-forget)
    import asyncio
    asyncio.ensure_future(
        guardar_consulta(solicitud.zona_id, solicitud.pregunta, respuesta.get("respuesta", ""))
    )

    return {
        "zona_id": solicitud.zona_id,
        "pregunta": solicitud.pregunta,
        **respuesta,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router_ia.get("/diagnostico-keys")
async def diagnostico_keys():
    """Verifica qué API keys están configuradas."""
    import os
    return {
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "DATABASE_URL": bool(os.getenv("DATABASE_URL")),
    }


@router_ia.get("/consultar/{zona_id}")
async def consultar_get(zona_id: str, q: str):
    """Versión GET para consultas rápidas desde el navegador. Ej: ?q=cuando+escampa"""
    solicitud = SolicitudConsulta(pregunta=q, zona_id=zona_id)
    return await consultar(solicitud)
