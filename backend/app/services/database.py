"""
Servicio de Base de Datos — PostgreSQL con asyncpg.
100% opcional: la app funciona completa sin PostgreSQL.
Todas las operaciones son fire-and-forget (no bloquean respuestas).
"""
import asyncio
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

import asyncpg

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None

_raw_url = os.getenv("DATABASE_URL", "postgresql://omaira:riesgo123@localhost:5432/riesgo_antioquia")
# asyncpg requiere postgresql://, Railway a veces entrega postgres://
DB_URL = _raw_url.replace("postgres://", "postgresql://", 1)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS zonas (
    id SERIAL PRIMARY KEY,
    zona_id VARCHAR(100) UNIQUE NOT NULL,
    municipio VARCHAR(200) NOT NULL,
    departamento VARCHAR(100) DEFAULT 'Antioquia',
    radio_km FLOAT DEFAULT 25,
    activa BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predicciones (
    id SERIAL PRIMARY KEY,
    zona_id VARCHAR(100) NOT NULL,
    tipo_riesgo VARCHAR(50) NOT NULL,
    horizonte VARCHAR(10) NOT NULL,
    nivel VARCHAR(20) NOT NULL,
    probabilidad FLOAT NOT NULL,
    amenaza FLOAT,
    exposicion FLOAT,
    vulnerabilidad FLOAT,
    factor_clima FLOAT,
    riesgo_total FLOAT,
    modo_degradado BOOLEAN DEFAULT FALSE,
    timestamp_prediccion TIMESTAMP NOT NULL,
    timestamp_horizonte TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS alertas (
    id SERIAL PRIMARY KEY,
    alerta_id VARCHAR(200) UNIQUE NOT NULL,
    zona_id VARCHAR(100),
    tipo_riesgo VARCHAR(50) NOT NULL,
    nivel VARCHAR(20) NOT NULL,
    municipio VARCHAR(200),
    descripcion TEXT,
    acciones TEXT,
    activa BOOLEAN DEFAULT TRUE,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS consultas_ia (
    id SERIAL PRIMARY KEY,
    zona_id VARCHAR(100),
    pregunta TEXT NOT NULL,
    respuesta TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

INSERT INTO zonas (zona_id, municipio) VALUES
    ('guatape', 'Guatapé'),
    ('medellin', 'Medellín'),
    ('rionegro', 'Rionegro'),
    ('santa_fe_antioquia', 'Santa Fe de Antioquia'),
    ('caucasia', 'Caucasia')
ON CONFLICT (zona_id) DO NOTHING;
"""


async def _connect_db() -> None:
    global _pool
    try:
        pool = await asyncio.wait_for(
            asyncpg.create_pool(DB_URL, min_size=1, max_size=5, command_timeout=10),
            timeout=10,
        )
        async with pool.acquire() as conn:
            await conn.execute(_SCHEMA)
        _pool = pool
        logger.info("PostgreSQL conectado — histórico de predicciones activo")
    except Exception as e:
        logger.warning(f"PostgreSQL no disponible ({type(e).__name__}) — app funciona sin histórico")
        _pool = None


async def init_pool() -> None:
    """Lanza la conexión en background — el startup no se bloquea."""
    asyncio.ensure_future(_connect_db())


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _pool_disponible() -> bool:
    return _pool is not None


# ── Escritura — fire-and-forget ───────────────────────────────────────────────

async def guardar_predicciones(zona_id: str, predicciones: List[Dict]) -> None:
    """
    Guarda predicciones en la tabla `predicciones`.
    Se llama desde riesgo_service sin bloquear la respuesta HTTP.
    """
    if not _pool_disponible():
        return
    rows = []
    for p in predicciones:
        comp = p.get("componentes", {})
        rows.append((
            zona_id,
            p.get("tipo_riesgo"),
            p.get("horizonte"),
            p.get("nivel"),
            p.get("probabilidad", 0),
            comp.get("amenaza", 0),
            comp.get("exposicion", 0),
            comp.get("vulnerabilidad", 0),
            comp.get("factor_clima", 1),
            comp.get("riesgo_total", 0),
            p.get("modo_degradado", False),
            _parse_ts(p.get("timestamp_prediccion")),
            _parse_ts(p.get("timestamp_horizonte")),
        ))
    try:
        async with _pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO predicciones
                   (zona_id, tipo_riesgo, horizonte, nivel, probabilidad,
                    amenaza, exposicion, vulnerabilidad, factor_clima, riesgo_total,
                    modo_degradado, timestamp_prediccion, timestamp_horizonte)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)""",
                rows,
            )
    except Exception as e:
        logger.debug(f"guardar_predicciones: {e}")


async def guardar_alerta(zona_id: str, alerta: Dict) -> None:
    """Guarda una alerta en la tabla `alertas`."""
    if not _pool_disponible():
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO alertas
                   (alerta_id, zona_id, tipo_riesgo, nivel, municipio, descripcion, acciones, timestamp)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                   ON CONFLICT (alerta_id) DO NOTHING""",
                alerta.get("alerta_id"),
                zona_id,
                alerta.get("tipo_riesgo"),
                alerta.get("nivel"),
                alerta.get("municipio"),
                alerta.get("descripcion"),
                str(alerta.get("acciones", [])),
                _parse_ts(alerta.get("timestamp")) or datetime.utcnow(),
            )
    except Exception as e:
        logger.debug(f"guardar_alerta: {e}")


async def guardar_consulta(zona_id: str, pregunta: str, respuesta: str) -> None:
    """Guarda preguntas y respuestas para mejora continua."""
    if not _pool_disponible():
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO consultas_ia (zona_id, pregunta, respuesta, timestamp)
                   VALUES ($1,$2,$3,$4)""",
                zona_id, pregunta[:500], respuesta[:2000], datetime.utcnow(),
            )
    except Exception as e:
        logger.debug(f"guardar_consulta: {e}")


# ── Lectura — histórico ───────────────────────────────────────────────────────

async def get_historico_predicciones(
    zona_id: str,
    tipo_riesgo: Optional[str] = None,
    horas: int = 24,
) -> List[Dict]:
    """
    Retorna predicciones históricas de las últimas N horas.
    Devuelve lista vacía si DB no disponible.
    """
    if not _pool_disponible():
        return []
    filtro_tipo = "AND tipo_riesgo = $3" if tipo_riesgo else ""
    params = [zona_id, horas, tipo_riesgo] if tipo_riesgo else [zona_id, horas]
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT tipo_riesgo, horizonte, nivel, probabilidad,
                           riesgo_total, timestamp_prediccion
                    FROM predicciones
                    WHERE zona_id = $1
                      AND timestamp_prediccion > NOW() - INTERVAL '$2 hours'
                    {filtro_tipo}
                    ORDER BY timestamp_prediccion DESC
                    LIMIT 200""",
                *params,
            )
            return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"get_historico: {e}")
        return []


async def get_stats_zona(zona_id: str) -> Dict:
    """Estadísticas básicas de una zona para el dashboard."""
    if not _pool_disponible():
        return {"disponible": False}
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT
                     COUNT(*) as total_predicciones,
                     COUNT(DISTINCT tipo_riesgo) as tipos_monitoreados,
                     MAX(timestamp_prediccion) as ultima_prediccion,
                     AVG(probabilidad) as probabilidad_media,
                     SUM(CASE WHEN nivel IN ('alto','muy_alto','critico') THEN 1 ELSE 0 END) as eventos_altos
                   FROM predicciones
                   WHERE zona_id = $1
                     AND timestamp_prediccion > NOW() - INTERVAL '7 days'""",
                zona_id,
            )
            return {
                "disponible": True,
                "zona_id": zona_id,
                **{k: (str(v) if isinstance(v, datetime) else v) for k, v in dict(row).items()},
            }
    except Exception as e:
        logger.debug(f"get_stats: {e}")
        return {"disponible": False}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
