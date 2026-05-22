"""
Servicio de Consultas en Lenguaje Natural.
Usa Gemini (gratis) si GEMINI_API_KEY está disponible, si no usa motor local.
"""
import asyncio
import httpx
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

from app.services.openmeteo_service import obtener_meteo_real, COORDS_ZONAS, nombre_zona

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

TZ_COL = ZoneInfo("America/Bogota")

# Códigos WMO que indican lluvia/precipitación
CODIGOS_LLUVIA = set(range(51, 68)) | set(range(80, 83)) | set(range(95, 100)) | {61, 63, 65, 66, 67}

# Descripción amigable de códigos WMO
DESCRIPCIONES_WMO = {
    0: "cielo despejado", 1: "mayormente despejado", 2: "parcialmente nublado", 3: "nublado",
    45: "niebla", 48: "niebla con escarcha",
    51: "llovizna ligera", 53: "llovizna moderada", 55: "llovizna intensa",
    61: "lluvia ligera", 63: "lluvia moderada", 65: "lluvia intensa",
    66: "lluvia con hielo", 67: "lluvia con hielo fuerte",
    71: "nevada ligera", 73: "nevada moderada", 75: "nevada intensa",
    80: "chubascos ligeros", 81: "chubascos moderados", 82: "chubascos intensos",
    95: "tormenta eléctrica", 96: "tormenta con granizo", 99: "tormenta con granizo fuerte",
}


async def _responder_con_gemini(pregunta: str, zona_id: str, lat: float, lon: float, zona_nombre: str) -> Optional[Dict]:
    """Usa Gemini para responder cualquier pregunta en lenguaje natural."""
    api_key = os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
    if not api_key:
        return None
    try:
        meteo = await obtener_meteo_real(zona_id, lat, lon)
        try:
            pronostico = await _fetch_pronostico_horario(lat, lon, dias=2)
            tiempos = pronostico.get("time", [])
            precips = pronostico.get("precipitation", [])
            probs = pronostico.get("precipitation_probability", [])
            prox_horas = []
            ahora = datetime.now(TZ_COL)
            idx = next((i for i, t in enumerate(tiempos) if t[:13] == ahora.strftime("%Y-%m-%dT%H")), 0)
            for i in range(idx, min(idx + 12, len(tiempos))):
                prox_horas.append(f"  {tiempos[i][11:16]}: {precips[i] if i<len(precips) else 0:.1f}mm ({probs[i] if i<len(probs) else 0}%)")
            pronostico_txt = "\n".join(prox_horas[:8])
        except Exception:
            pronostico_txt = "no disponible"

        contexto = f"""Eres OMAIRA, sistema experto en gestión de riesgo ambiental para Antioquia, Colombia.
Respondes en español, de forma concisa y útil (máx 3 oraciones).

ZONA: {zona_nombre} ({zona_id})
FECHA/HORA: {datetime.now(TZ_COL).strftime('%Y-%m-%d %H:%M')} (hora Colombia)

DATOS METEOROLÓGICOS ACTUALES (Open-Meteo):
- Temperatura: {meteo.get('temperatura_c', 'N/D')}°C
- Humedad relativa: {meteo.get('humedad_relativa', 'N/D')}%
- Precipitación actual: {meteo.get('precipitacion_actual_mm', 0)} mm/h
- Lluvia acumulada 24h: {meteo.get('lluvia_24h_mm', 0):.1f} mm
- Viento: {meteo.get('velocidad_viento_ms', 0)} m/s
- Condición: {DESCRIPCIONES_WMO.get(meteo.get('codigo_clima'), 'sin datos')}
- Humedad suelo: {meteo.get('humedad_suelo', 0)*100:.0f}%

PRONÓSTICO PRÓXIMAS HORAS (precipitación/probabilidad):
{pronostico_txt}

Responde esta pregunta del usuario de forma directa y útil:
{pregunta}"""

        payload = {"contents": [{"parts": [{"text": contexto}]}]}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{GEMINI_URL}?key={api_key}",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if r.status_code != 200:
                return {"pregunta_tipo": "gemini_error", "respuesta": f"Gemini HTTP {r.status_code}: {r.text[:300]}", "datos": {}}
            data = r.json()
            texto = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {
                "pregunta_tipo": "gemini",
                "respuesta": texto,
                "datos": {"fuente": "Gemini + Open-Meteo", "zona": zona_nombre}
            }
    except Exception as e:
        logger.warning(f"Gemini error: {type(e).__name__}: {e}")
        return {
            "pregunta_tipo": "gemini_error",
            "respuesta": f"[Gemini error: {type(e).__name__}: {str(e)[:200]}]",
            "datos": {"fuente": "error"}
        }


async def responder_consulta(
    pregunta: str,
    zona_id: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> Dict:
    """
    Punto de entrada principal.
    Detecta el tipo de pregunta y retorna una respuesta estructurada.
    """
    lat = lat or COORDS_ZONAS.get(zona_id, (6.2336, -75.1567))[0]
    lon = lon or COORDS_ZONAS.get(zona_id, (6.2336, -75.1567))[1]
    pregunta_lower = pregunta.lower().strip()
    zona_nombre = nombre_zona(zona_id)

    # Gemini como primera opción si hay API key
    gemini_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if gemini_key:
        resultado = await _responder_con_gemini(pregunta, zona_id, lat, lon, zona_nombre)
        if resultado:
            return resultado
        # Gemini no disponible — continúa con motor local

    # Clasificación local como fallback
    if _es_pregunta_escampar(pregunta_lower):
        return await _cuando_escampa(zona_id, lat, lon, zona_nombre)
    elif _es_pregunta_va_a_llover(pregunta_lower):
        return await _va_a_llover(zona_id, lat, lon, zona_nombre)
    elif _es_pregunta_temperatura(pregunta_lower):
        return await _pronostico_temperatura(zona_id, lat, lon, zona_nombre)
    elif _es_pregunta_viento(pregunta_lower):
        return await _pronostico_viento(zona_id, lat, lon, zona_nombre)
    elif _es_pregunta_riesgo(pregunta_lower):
        return await _resumen_riesgo(zona_id, lat, lon, zona_nombre)
    elif _es_pregunta_condicion_actual(pregunta_lower):
        return await _condicion_actual(zona_id, lat, lon, zona_nombre)
    else:
        return await _respuesta_general(pregunta, zona_id, lat, lon, zona_nombre)


# ── Clasificadores ─────────────────────────────────────────────────────────────

def _es_pregunta_escampar(q: str) -> bool:
    palabras = ["escampa", "escampara", "escampará", "para de llover", "deja de llover",
                "cuando para", "cuándo para", "cuanto tarda", "cuánto tarda",
                "cuanto tiempo llueve", "cuando termina la lluvia", "deja de llover",
                "termina la lluvia", "para la lluvia", "se va la lluvia"]
    return any(p in q for p in palabras)

def _es_pregunta_va_a_llover(q: str) -> bool:
    palabras = ["va a llover", "va llover", "lloverá", "llovera", "lluvia hoy",
                "pronóstico lluvia", "pronostico lluvia", "llueve hoy",
                "habrá lluvia", "habra lluvia", "cuando llueve", "cuando lluvia",
                "cuando llover", "lloverá hoy", "llovera hoy", "va llover",
                "va a caer lluvia", "caerá lluvia", "caera lluvia"]
    return any(p in q for p in palabras)

def _es_pregunta_temperatura(q: str) -> bool:
    palabras = ["temperatura", "calor", "frío", "frio", "cuánto hace", "cuanto hace",
                "que tan caliente", "grados", "°c", "que temperatura", "qué temperatura",
                "hace frio", "hace calor", "esta frio", "está frío"]
    return any(p in q for p in palabras)

def _es_pregunta_viento(q: str) -> bool:
    palabras = ["viento", "ventoso", "brisa", "ráfaga", "rafaga"]
    return any(p in q for p in palabras)

def _es_pregunta_riesgo(q: str) -> bool:
    palabras = ["riesgo", "peligro", "seguro", "deslizamiento", "inundación",
                "inundacion", "derrumbe", "alerta"]
    return any(p in q for p in palabras)

def _es_pregunta_condicion_actual(q: str) -> bool:
    palabras = ["clima ahora", "clima actual", "tiempo ahora", "qué tiempo",
                "que tiempo", "cómo está el tiempo", "como esta el tiempo",
                "condiciones actuales", "está lloviendo", "esta lloviendo"]
    return any(p in q for p in palabras)


# ── Respuestas específicas ─────────────────────────────────────────────────────

async def _cuando_escampa(zona_id: str, lat: float, lon: float, zona_nombre: str) -> Dict:
    """
    Calcula cuándo dejará de llover basándose en el pronóstico horario.
    """
    try:
        pronostico = await _fetch_pronostico_horario(lat, lon, dias=2)
    except Exception:
        return _respuesta_sin_datos("cuando escampa", zona_nombre)

    ahora = datetime.now(TZ_COL)
    hora_actual_str = ahora.strftime("%Y-%m-%dT%H:00")

    tiempos = pronostico["time"]
    precipitaciones = pronostico["precipitation"]
    probs = pronostico.get("precipitation_probability", [0] * len(tiempos))
    codigos = pronostico.get("weather_code", [0] * len(tiempos))

    # Encontrar índice de la hora actual
    idx_ahora = next(
        (i for i, t in enumerate(tiempos) if t[:13] == hora_actual_str[:13]),
        0
    )

    lluvia_ahora = precipitaciones[idx_ahora] if idx_ahora < len(precipitaciones) else 0
    codigo_ahora = codigos[idx_ahora] if idx_ahora < len(codigos) else 0

    # Si no está lloviendo ahora
    if lluvia_ahora < 0.1 and codigo_ahora not in CODIGOS_LLUVIA:
        return {
            "pregunta_tipo": "cuando_escampa",
            "respuesta": f"En {zona_nombre} no está lloviendo en este momento. "
                         f"El cielo está {DESCRIPCIONES_WMO.get(codigo_ahora, 'despejado')}.",
            "detalle": _proximo_aguacero(tiempos, precipitaciones, codigos, probs, idx_ahora, ahora),
            "datos": {
                "lluvia_actual_mm": round(lluvia_ahora, 1),
                "condicion_actual": DESCRIPCIONES_WMO.get(codigo_ahora, "sin datos"),
                "fuente": "Open-Meteo pronóstico"
            }
        }

    # Buscar cuándo deja de llover (primer bloque de ≥2h seguidas sin lluvia)
    hora_escampa = None
    horas_seguidas_secos = 0
    for i in range(idx_ahora + 1, min(idx_ahora + 49, len(tiempos))):
        precip = precipitaciones[i] if i < len(precipitaciones) else 0
        cod = codigos[i] if i < len(codigos) else 0
        es_seco = precip < 0.2 and cod not in CODIGOS_LLUVIA
        if es_seco:
            horas_seguidas_secos += 1
            if horas_seguidas_secos >= 2:
                hora_escampa = i - 1
                break
        else:
            horas_seguidas_secos = 0

    if hora_escampa is None:
        return {
            "pregunta_tipo": "cuando_escampa",
            "respuesta": f"La lluvia en {zona_nombre} parece persistir durante las próximas 48 horas "
                         f"según el pronóstico de Open-Meteo. Se recomienda precaución vial.",
            "datos": {"lluvia_actual_mm": round(lluvia_ahora, 1), "fuente": "Open-Meteo"}
        }

    # Calcular tiempo restante
    ts_escampa = datetime.fromisoformat(tiempos[hora_escampa]).replace(tzinfo=TZ_COL)
    delta = ts_escampa - ahora
    horas_restantes = delta.total_seconds() / 3600
    minutos_restantes = int((delta.total_seconds() % 3600) / 60)

    if horas_restantes < 0.5:
        tiempo_texto = f"en aproximadamente {int(delta.total_seconds() / 60)} minutos"
    elif horas_restantes < 1.5:
        tiempo_texto = f"en aproximadamente 1 hora ({ts_escampa.strftime('%H:%M')})"
    else:
        tiempo_texto = (
            f"en aproximadamente {int(horas_restantes)} horas y {minutos_restantes} minutos "
            f"(alrededor de las {ts_escampa.strftime('%H:%M')})"
        )

    # Lluvia total hasta que escampe
    lluvia_total = sum(
        precipitaciones[i] for i in range(idx_ahora, hora_escampa + 1)
        if i < len(precipitaciones)
    )

    respuesta = (
        f"Según el pronóstico de Open-Meteo, la lluvia en {zona_nombre} debería escampar "
        f"{tiempo_texto}. "
        f"Se esperan {lluvia_total:.1f} mm acumulados hasta entonces."
    )

    return {
        "pregunta_tipo": "cuando_escampa",
        "respuesta": respuesta,
        "hora_estimada_escampar": ts_escampa.strftime("%H:%M"),
        "horas_restantes": round(horas_restantes, 1),
        "lluvia_acumulada_mm": round(lluvia_total, 1),
        "datos": {
            "lluvia_actual_mm": round(lluvia_ahora, 1),
            "condicion_actual": DESCRIPCIONES_WMO.get(codigo_ahora, "lluvia"),
            "fuente": "Open-Meteo pronóstico horario",
        }
    }


async def _va_a_llover(zona_id: str, lat: float, lon: float, zona_nombre: str) -> Dict:
    """Pronóstico de lluvia para las próximas horas."""
    try:
        pronostico = await _fetch_pronostico_horario(lat, lon, dias=1)
    except Exception:
        return _respuesta_sin_datos("va a llover", zona_nombre)

    ahora = datetime.now(TZ_COL)
    tiempos = pronostico["time"]
    precipitaciones = pronostico["precipitation"]
    probs = pronostico.get("precipitation_probability", [])
    codigos = pronostico.get("weather_code", [])

    idx_ahora = next((i for i, t in enumerate(tiempos) if t[:13] == ahora.strftime("%Y-%m-%dT%H")), 0)

    # Próximas 12 horas
    ventana = 12
    resumen_horas = []
    lluvia_total_12h = 0
    for i in range(idx_ahora, min(idx_ahora + ventana, len(tiempos))):
        precip = precipitaciones[i] if i < len(precipitaciones) else 0
        prob = probs[i] if i < len(probs) else 0
        cod = codigos[i] if i < len(codigos) else 0
        hora_str = tiempos[i][11:16] if i < len(tiempos) else "?"
        lluvia_total_12h += precip
        if precip > 0.2 or cod in CODIGOS_LLUVIA:
            resumen_horas.append(f"  {hora_str}: {DESCRIPCIONES_WMO.get(cod, 'lluvia')} ({precip:.1f}mm, {prob}% prob.)")

    if not resumen_horas:
        respuesta = f"No se espera lluvia significativa en {zona_nombre} durante las próximas 12 horas. ☀️"
    else:
        respuesta = (
            f"Sí, se espera lluvia en {zona_nombre} en las próximas horas. "
            f"Total estimado: {lluvia_total_12h:.1f} mm en 12 horas.\n"
            f"Horas con precipitación:\n" + "\n".join(resumen_horas[:6])
        )

    return {
        "pregunta_tipo": "pronostico_lluvia",
        "respuesta": respuesta,
        "lluvia_total_12h_mm": round(lluvia_total_12h, 1),
        "horas_con_lluvia": len(resumen_horas),
        "datos": {"fuente": "Open-Meteo pronóstico"}
    }


async def _pronostico_temperatura(zona_id: str, lat: float, lon: float, zona_nombre: str) -> Dict:
    meteo = await obtener_meteo_real(zona_id, lat, lon)
    temp = meteo.get("temperatura_c", "N/D")
    hum = meteo.get("humedad_relativa", "N/D")
    return {
        "pregunta_tipo": "temperatura",
        "respuesta": (
            f"En {zona_nombre} la temperatura actual es {temp}°C con {hum}% de humedad relativa. "
            f"{'Se siente húmedo y fresco.' if hum and hum > 80 else 'Condiciones normales para la región.'}"
        ),
        "datos": {"temperatura_c": temp, "humedad_pct": hum, "fuente": meteo.get("fuente")}
    }


async def _pronostico_viento(zona_id: str, lat: float, lon: float, zona_nombre: str) -> Dict:
    meteo = await obtener_meteo_real(zona_id, lat, lon)
    viento = meteo.get("velocidad_viento_ms", 0)
    nivel = "calmo" if viento < 2 else "leve" if viento < 5 else "moderado" if viento < 10 else "fuerte"
    return {
        "pregunta_tipo": "viento",
        "respuesta": f"En {zona_nombre} el viento está {nivel}: {viento} m/s ({viento*3.6:.0f} km/h).",
        "datos": {"velocidad_ms": viento, "nivel": nivel, "fuente": meteo.get("fuente")}
    }


async def _condicion_actual(zona_id: str, lat: float, lon: float, zona_nombre: str) -> Dict:
    meteo = await obtener_meteo_real(zona_id, lat, lon)
    cod = meteo.get("codigo_clima")
    descripcion = DESCRIPCIONES_WMO.get(cod, "sin datos") if cod is not None else "sin datos"
    temp = meteo.get("temperatura_c", "N/D")
    precip = meteo.get("precipitacion_actual_mm", 0)
    hum = meteo.get("humedad_relativa", "N/D")
    viento = meteo.get("velocidad_viento_ms", 0)

    llueve_txt = f" Precipitación actual: {precip} mm/h." if precip and precip > 0.1 else ""
    return {
        "pregunta_tipo": "condicion_actual",
        "respuesta": (
            f"En {zona_nombre} hay {descripcion}. "
            f"Temperatura: {temp}°C | Humedad: {hum}% | Viento: {viento} m/s.{llueve_txt}"
        ),
        "datos": meteo
    }


async def _resumen_riesgo(zona_id: str, lat: float, lon: float, zona_nombre: str) -> Dict:
    from app.services.riesgo_service import calcular_riesgo_zona
    resultado = await calcular_riesgo_zona(zona_id)
    resumen = resultado.get("resumen", {})
    nivel = resumen.get("nivel_maximo", "bajo")
    tipo_dom = resumen.get("riesgo_dominante", "deslizamiento")
    prob = resumen.get("probabilidad_maxima", 0)

    emojis = {"muy_bajo": "🟢", "bajo": "🟡", "medio": "🟠", "alto": "🔴", "muy_alto": "🔴", "critico": "⛔"}
    emoji = emojis.get(nivel, "⚠️")

    respuesta = (
        f"{emoji} El nivel de riesgo actual en {zona_nombre} es {nivel.replace('_',' ').upper()}. "
        f"El riesgo dominante es {tipo_dom} con {prob*100:.0f}% de probabilidad. "
    )
    if nivel in ("alto", "muy_alto", "critico"):
        respuesta += "Se recomienda precaución y seguir las indicaciones de las autoridades locales."
    else:
        respuesta += "No hay alertas críticas activas en este momento."

    return {
        "pregunta_tipo": "riesgo",
        "respuesta": respuesta,
        "datos": {"nivel": nivel, "riesgo_dominante": tipo_dom, "probabilidad": prob}
    }


async def _respuesta_general(pregunta: str, zona_id: str, lat: float, lon: float, zona_nombre: str) -> Dict:
    """Para preguntas que no encajan en categorías específicas, usa el motor IA local."""
    from app.services.riesgo_service import calcular_riesgo_zona
    from app.services.irg_service import calcular_irg
    from app.services.ia_service import analizar_con_ia, ModeloIA
    from app.services.openmeteo_service import obtener_meteo_real

    meteo = await obtener_meteo_real(zona_id, lat, lon)
    irg_resultado = calcular_irg(meteo, {}, zona_id=zona_id)
    preds = await calcular_riesgo_zona(zona_id)

    datos_combinados = {
        "zona": zona_id,
        "sensores": {
            "lluvia24": meteo.get("lluvia_24h_mm", 0),
            "embalse": meteo.get("nivel_embalse_pct", 70),
            "temp": meteo.get("temperatura_c", 19),
            "hum": meteo.get("humedad_relativa", 70),
            "viento": meteo.get("velocidad_viento_ms", 3),
        },
        "irg": {"irg": irg_resultado.irg, "nivel": irg_resultado.nivel, "top_factores": irg_resultado.top_factores},
        "alertas_irg": irg_resultado.alertas_irg,
        "predicciones": preds.get("predicciones", []),
        "contexto_local": irg_resultado.contexto_local,
        "pregunta_usuario": pregunta,
    }

    analisis = await analizar_con_ia(datos_combinados, ModeloIA.LOCAL, lat=lat, lon=lon)
    return {
        "pregunta_tipo": "general",
        "respuesta": analisis.get("diagnostico", "No pude procesar tu pregunta."),
        "pronostico": analisis.get("pronostico"),
        "recomendaciones": analisis.get("recomendaciones", []),
        "datos": {"irg": irg_resultado.irg, "nivel": irg_resultado.nivel}
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _fetch_pronostico_horario(lat: float, lon: float, dias: int = 2) -> Dict:
    """Trae pronóstico horario de Open-Meteo (sin caché — necesita datos frescos)."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=precipitation,precipitation_probability,weather_code"
        f"&forecast_days={dias}"
        "&timezone=America%2FBogota"
    )
    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise ConnectionError(f"Open-Meteo HTTP {resp.status_code}")
        return resp.json().get("hourly", {})


def _proximo_aguacero(
    tiempos: List, precipitaciones: List, codigos: List, probs: List, idx: int, ahora: datetime
) -> Optional[str]:
    """Detecta el próximo aguacero si no está lloviendo ahora."""
    for i in range(idx + 1, min(idx + 25, len(tiempos))):
        precip = precipitaciones[i] if i < len(precipitaciones) else 0
        cod = codigos[i] if i < len(codigos) else 0
        prob = probs[i] if i < len(probs) else 0
        if precip > 0.5 or cod in CODIGOS_LLUVIA:
            ts = datetime.fromisoformat(tiempos[i]).replace(tzinfo=TZ_COL)
            delta_h = (ts - ahora).total_seconds() / 3600
            return f"El próximo aguacero se espera en ~{int(delta_h)}h ({ts.strftime('%H:%M')}, {prob}% probabilidad)."
    return "No se prevé lluvia significativa en las próximas 24 horas."


def _respuesta_sin_datos(tipo: str, zona_nombre: str) -> Dict:
    return {
        "pregunta_tipo": tipo,
        "respuesta": f"No pude obtener el pronóstico meteorológico para {zona_nombre} en este momento. "
                     "Verifica tu conexión a internet o intenta de nuevo.",
        "datos": {"fuente": "sin datos"}
    }
