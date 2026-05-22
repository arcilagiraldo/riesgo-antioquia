"""
Autenticación con Google OAuth — verificación de token y lista de emails autorizados.
"""
import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

GOOGLE_TOKEN_INFO_URL = "https://oauth2.googleapis.com/tokeninfo"

def _emails_autorizados() -> list[str]:
    raw = os.getenv("AUTHORIZED_EMAILS", "arcilagiraldo@gmail.com")
    return [e.strip().lower() for e in raw.split(",") if e.strip()]


class TokenRequest(BaseModel):
    credential: str  # JWT token de Google Identity Services


@router.post("/verify")
async def verify_google_token(req: TokenRequest):
    """
    Verifica el token de Google y comprueba si el email está autorizado.
    Retorna {ok: true, email, nombre} o lanza 403.
    """
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(f"{GOOGLE_TOKEN_INFO_URL}?id_token={req.credential}")
            if r.status_code != 200:
                raise HTTPException(401, "Token de Google inválido")
            info = r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error verificando token: {e}")

    email = info.get("email", "").lower()
    if not email:
        raise HTTPException(401, "Token sin email")

    if email not in _emails_autorizados():
        raise HTTPException(403, f"Acceso no autorizado para {email}")

    return {
        "ok": True,
        "email": email,
        "nombre": info.get("name", email.split("@")[0]),
        "foto": info.get("picture", ""),
    }


@router.get("/me")
async def check_session():
    """Endpoint de prueba — la sesión real se guarda en el frontend."""
    return {"status": "auth-activo"}
