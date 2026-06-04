import os
import asyncio
import uuid
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator
from dotenv import load_dotenv

# Cargamos las variables de entorno desde el archivo .env si existe localmente
load_dotenv()

# Intentamos importar la función de creación del agente usando rutas relativas y absolutas
# Esto asegura que funcione tanto si se ejecuta desde la raíz como dentro de la carpeta 'app/'
try:
    from agent import create_agent
except ImportError:
    from app.agent import create_agent

# =====================================================================
# 1. INICIALIZACIÓN DE LA APLICACIÓN FASTAPI
# =====================================================================

app = FastAPI(
    title="Webhook Agente Kubernetes/OpenShift",
    description="Webhook de FastAPI para integrar Open WebUI con un Agente SRE usando LangGraph.",
    version="1.0.0"
)

# Configuración de CORS (Cross-Origin Resource Sharing)
# Permite que la API reciba peticiones desde cualquier origen (e.g., el frontend de Open WebUI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global que almacenará el grafo compilado del agente
agent_executor = None

@app.on_event("startup")
async def startup_event():
    """
    Evento de ciclo de vida de FastAPI que se ejecuta al arrancar el servidor.
    Se utiliza para inicializar y compilar el flujo del agente LangGraph una sola vez
    y mantenerlo en memoria para optimizar el rendimiento.
    """
    global agent_executor
    print("[STARTUP] Iniciando el webhook del Agente SRE...")
    agent_executor = create_agent()
    print("[SUCCESS] Grafo de LangGraph compilado y listo para recibir consultas.")


# =====================================================================
# 2. DEFINICIÓN DE MODELOS DE PETICIÓN (Pydantic)
# =====================================================================

class WebhookRequest(BaseModel):
    """
    Representa el esquema del cuerpo de la petición JSON que se recibe desde Open WebUI.
    """
    url: str = Field(
        ...,
        description="URL que debe ser validada de forma asíncrona antes de iniciar el flujo."
    )
    message: str = Field(
        ...,
        description="Mensaje o pregunta enviada por el usuario al agente."
    )
    session_id: Optional[str] = Field(
        default="webhook-session-default",
        description="Identificador único de conversación para persistir el historial (AgentState) en LangGraph."
    )


# =====================================================================
# 3. VALIDADOR DE URL ASÍNCRONO
# =====================================================================

async def validate_url_async(url: str, timeout_seconds: float) -> bool:
    """
    Realiza una petición HTTP asíncrona hacia la URL especificada para validar su disponibilidad.
    
    Espera recibir un estado HTTP 200 OK. Cualquier otra respuesta (códigos 4xx, 5xx) o 
    problema de red (timeout, DNS fallido, etc.) se interpreta como una URL no disponible.
    
    Args:
        url (str): La URL que se desea comprobar.
        timeout_seconds (float): Tiempo de espera máximo en segundos.
        
    Returns:
        bool: True si la URL respondió con HTTP 200, False en cualquier otro caso.
    """
    # Usamos httpx.AsyncClient para no bloquear el bucle de eventos asíncronos de FastAPI
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            # Hacemos una petición GET. En algunos casos HEAD es más eficiente,
            # pero GET es más robusto para servidores que no implementen el método HEAD.
            response = await client.get(url)
            print(f"DEBUG: Validación de URL: {url} -> Código de respuesta: {response.status_code}")
            return response.status_code == 200
    except httpx.RequestError as exc:
        print(f"[ERROR] Error de red al validar la URL {url}: {exc}")
        return False
    except Exception as exc:
        print(f"[ERROR] Error inesperado al validar la URL {url}: {exc}")
        return False


# =====================================================================
# 4. GENERADOR SSE (Server-Sent Events) PARA STREAMING
# =====================================================================

async def sse_stream_generator(message: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Generador asíncrono que consume el flujo de eventos de LangGraph (astream_events)
    y produce fragmentos de texto con el formato estándar de Server-Sent Events (SSE).
    
    Formato SSE producido:
        data: <contenido_del_token>\n\n
        
    Args:
        message (str): Mensaje enviado por el usuario.
        session_id (str): Identificador de la conversación para cargar el historial de mensajes.
        
    Yields:
        str: Línea de texto formateada para SSE.
    """
    if agent_executor is None:
        yield "data: [ERROR: El agente de LangGraph no ha sido inicializado correctamente]\n\n"
        return

    # Configuramos la ejecución del grafo
    # El thread_id es usado por MemorySaver de LangGraph para identificar la conversación
    # y así recuperar el historial (AgentState) correspondiente.
    config = {"configurable": {"thread_id": session_id}}
    
    # Preparamos las entradas para el flujo del agente
    inputs = {
        "messages": [("user", message)]
    }

    try:
        # Usamos astream_events para obtener un flujo detallado de todo el ciclo del grafo.
        # Filtramos los eventos generados por el LLM en tiempo real.
        # Versión "v2" de la API de streaming de LangChain.
        async for event in agent_executor.astream_events(inputs, config=config, version="v2"):
            # Buscamos eventos de tipo 'on_chat_model_stream' para capturar los tokens
            # que genera el LLM conforme van saliendo.
            if event["event"] == "on_chat_model_stream":
                # El token o fragmento está en la propiedad chunk
                chunk = event["data"]["chunk"]
                content = chunk.content
                if content:
                    # El protocolo SSE requiere enviar prefijado con 'data: ' y terminar con '\n\n'
                    yield f"data: {content}\n\n"
                    # Pequeña pausa para permitir que el event loop ceda el control si es necesario
                    await asyncio.sleep(0.01)
                    
    except Exception as e:
        print(f"[ERROR] Error durante el streaming del agente: {e}")
        yield f"data: [ERROR: Ocurrió un error en el flujo del modelo: {str(e)}]\n\n"


# =====================================================================
# 5. ENDPOINT DE WEBHOOK POST
# =====================================================================

@app.post("/webhook")
async def webhook_endpoint(request_data: WebhookRequest):
    """
    Webhook principal del sistema. Recibe peticiones HTTP POST con 'url' y 'message'.
    
    Flujo de la petición:
    1. Lee la configuración del timeout desde las variables de entorno.
    2. Valida la URL de manera asíncrona. Si no responde con HTTP 200, interrumpe
       el flujo inmediatamente y devuelve un error 400 Bad Request.
    3. Si la URL es válida, retorna una respuesta de transmisión (StreamingResponse)
       en formato Server-Sent Events (SSE) que contiene los tokens generados por el agente.
       
    Args:
        request_data (WebhookRequest): Payload JSON conteniendo url, message y opcionalmente session_id.
        
    Returns:
        StreamingResponse: Flujo SSE en vivo de la respuesta del agente.
    """
    # 1. Obtener timeout de validación de la URL desde las variables de entorno
    try:
        url_timeout = float(os.getenv("URL_VALIDATION_TIMEOUT", "5"))
    except ValueError:
        url_timeout = 5.0

    print(f"[INFO] Petición de Webhook recibida.")
    print(f"   > URL a validar: {request_data.url}")
    print(f"   > Mensaje: {request_data.message[:50]}...")
    print(f"   > Session ID: {request_data.session_id}")

    # 2. Validar URL asíncronamente
    is_url_valid = await validate_url_async(request_data.url, url_timeout)
    if not is_url_valid:
        print(f"[WARNING] Validación de URL fallida para {request_data.url}. Abortando petición con HTTP 400.")
        raise HTTPException(
            status_code=400,
            detail=f"La URL proporcionada '{request_data.url}' no respondió con un estado HTTP 200 OK. Flujo cancelado."
        )

    print("[SUCCESS] URL validada con éxito. Iniciando streaming del agente LangGraph...")
    
    # 3. Retornar respuesta en streaming con SSE
    # Indicamos el media_type "text/event-stream" que es el estándar para SSE
    return StreamingResponse(
        sse_stream_generator(request_data.message, request_data.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# =====================================================================
# 6. ENDPOINT DE VERIFICACIÓN DE SALUD (Health Check)
# =====================================================================

@app.get("/health")
async def health_check():
    """
    Ruta para la verificación del estado de salud de la aplicación (Liveness/Readiness probes).
    Útil para integraciones en entornos de Kubernetes y OpenShift.
    """
    return {
        "status": "ok",
        "agent_initialized": agent_executor is not None
    }


# =====================================================================
# 7. EJECUCIÓN LOCAL
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    # Leemos el puerto de escucha desde el entorno o usamos 8080 por defecto (estándar de OpenShift)
    try:
        port = int(os.getenv("AGENT_PORT", "8080"))
    except ValueError:
        port = 8080
        
    print(f"[INFO] Iniciando Uvicorn en el puerto {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
