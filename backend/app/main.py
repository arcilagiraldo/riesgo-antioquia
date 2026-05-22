"""OMAIRA v4 — Backend FastAPI"""
import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api import riesgo, alertas, sensores, prediccion, configuracion
from app.api.avanzado import router_irg, router_ia
from app.api.auth import router as router_auth
from app.services.websocket_manager import ConnectionManager
from app.services.riesgo_service import generar_alertas
from app.services.database import init_pool, close_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()   # intenta conectar a PostgreSQL (falla silenciosamente si no hay DB)
    yield
    await close_pool()


app = FastAPI(title="OMAIRA v4 API", version="4.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers REST ──────────────────────────────────────────────────────────────
app.include_router(riesgo.router,          prefix="/api/v1/riesgo")
app.include_router(alertas.router,         prefix="/api/v1/alertas")
app.include_router(sensores.router,        prefix="/api/v1/sensores")
app.include_router(prediccion.router,      prefix="/api/v1/prediccion")
app.include_router(router_irg,             prefix="/api/v1/irg")
app.include_router(router_ia,              prefix="/api/v1/ia")
app.include_router(configuracion.router,   prefix="/api/v1/config")
app.include_router(router_auth,            prefix="/api/v1/auth")

# ── WebSocket ─────────────────────────────────────────────────────────────────
manager = ConnectionManager()


@app.websocket("/ws/alertas")
async def ws_alertas(websocket: WebSocket):
    """Alertas en tiempo real — broadcast cada 30 segundos"""
    await manager.connect(websocket)
    try:
        while True:
            zonas = ["guatape", "medellin", "rionegro"]
            todas = []
            for zona in zonas:
                alertas_zona = await generar_alertas(zona)
                todas.extend(alertas_zona)
            await websocket.send_json({
                "tipo": "alertas",
                "total": len(todas),
                "alertas": todas,
            })
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/riesgo-live")
async def ws_riesgo_live(websocket: WebSocket):
    """Actualizaciones de riesgo en tiempo real — broadcast cada 60 segundos"""
    from app.services.riesgo_service import calcular_riesgo_zona
    await manager.connect(websocket)
    try:
        while True:
            datos = {}
            for zona in ["guatape", "medellin", "rionegro"]:
                r = await calcular_riesgo_zona(zona)
                datos[zona] = {
                    "municipio": r["municipio"],
                    "nivel_maximo": r["resumen"]["nivel_maximo"],
                    "riesgo_dominante": r["resumen"]["riesgo_dominante"],
                    "probabilidad_maxima": r["resumen"]["probabilidad_maxima"],
                }
            await websocket.send_json({"tipo": "riesgo_live", "zonas": datos})
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "OMAIRA v4 Backend activo", "version": "4.1.0"}


@app.get("/health")
def health():
    return {"status": "ok", "sistema": "OMAIRA v4"}
