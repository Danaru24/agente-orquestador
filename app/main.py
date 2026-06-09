import os
import asyncio
import uuid
import httpx
import traceback
import sys
import re
import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator
from contextlib import AsyncExitStack
from dotenv import load_dotenv

# Importaciones del cliente MCP oficial y adaptadores
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools

# Cargamos las variables de entorno desde el archivo .env si existe localmente
load_dotenv()

# Intentamos importar la función de creación del agente usando rutas relativas y absolutas
try:
    from agent import create_agent
except ImportError:
    from app.agent import create_agent

# =====================================================================
# CONFIGURACIÓN GLOBAL DE VARIABLES DE ENTORNO
# =====================================================================

# Lista de servidores MCP remotos para conexión simultánea
# Cada URL debe apuntar al endpoint base del servidor MCP (se redirige automáticamente a /sse)
MCP_SERVERS = [
    url.strip() for url in os.getenv(
        "MCP_SERVERS",
        "https://kubernetes-mcp-server-infra-ai.apps.ocp.zz987.sandbox2813.opentlc.com/mcp,https://zabbix-mcp-server-agente-command.apps.ocp.zz987.sandbox2813.opentlc.com/mcp"
    ).split(",") if url.strip()
]

# =====================================================================
# 1. INICIALIZACIÓN DE LA APLICACIÓN FASTAPI
# =====================================================================

app = FastAPI(
    title="Webhook Agente Kubernetes/OpenShift con MCP SSE",
    description="Webhook de FastAPI para integrar Open WebUI con un Agente SRE usando LangGraph y MCP SSE.",
    version="1.1.0"
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    Evento de ciclo de vida de FastAPI al arrancar el servidor.
    """
    print("[STARTUP] Iniciando el webhook del Agente SRE con soporte MCP...")
    for i, url in enumerate(MCP_SERVERS, 1):
        print(f"[STARTUP] Servidor MCP #{i}: {url}")
    print("[SUCCESS] Webhook listo para compilar agentes dinámicamente por petición.")


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
        description="Identificador único de conversación para persistir el historial en LangGraph."
    )


# =====================================================================
# 3. VALIDADOR DE URL ASÍNCRONO
# =====================================================================

async def validate_url_async(url: str, timeout_seconds: float) -> bool:
    """
    Realiza una petición HTTP asíncrona hacia la URL especificada para validar su disponibilidad.
    Espera recibir un estado HTTP 200 OK.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
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
# 4. AUXILIAR DE RENDERIZACIÓN SSE Y GENERADOR
# =====================================================================

def format_sse(text: str) -> str:
    """
    Formatea un texto según la especificación de Server-Sent Events (SSE).
    Cada línea del texto debe ir precedida por 'data: '. El mensaje completo debe terminar con '\n\n'.
    Esto previene que OpenWebUI pierda los saltos de línea internos de Markdown.
    """
    lines = text.split("\n")
    return "\n".join(f"data: {line}" for line in lines) + "\n\n"


async def sse_stream_generator(message: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Generador asíncrono que establece la conexión con el servidor MCP remoto vía SSE,
    inicializa la ClientSession, mapea dinámicamente las herramientas a LangChain,
    compila el grafo de LangGraph, y consume el stream de eventos (astream_events).
    
    Asegura que la sesión MCP permanezca abierta durante toda la ejecución de LangGraph
    al envolver el ciclo de vida del agente dentro de los bloques de contexto asíncronos.
    """
    config = {"configurable": {"thread_id": session_id}}
    inputs = {
        "messages": [("user", message)]
    }

    print(f"[INFO] Iniciando conexión con {len(MCP_SERVERS)} servidor(es) MCP...")

    def resolve_sse_url(url: str) -> str:
        """Convierte la URL del servidor MCP al endpoint SSE correcto."""
        if url.endswith("/mcp"):
            return url[:-4] + "/sse"
        elif not url.endswith("/sse"):
            return url.rstrip("/") + "/sse"
        return url

    try:
        async with AsyncExitStack() as stack:
            all_tools = []

            # Conectamos con cada servidor de la lista
            for url in MCP_SERVERS:
                sse_url = resolve_sse_url(url)
                try:
                    # 1. Abrimos conexión SSE
                    streams = await stack.enter_async_context(sse_client(url=sse_url, timeout=None))
                    read_stream, write_stream = streams

                    # 2. Inicializamos la sesión para este servidor
                    session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                    await session.initialize()
                    print(f"[SUCCESS] Sesión MCP inicializada para: {url}")

                    # 3. Descargamos las herramientas
                    server_tools = await load_mcp_tools(session)
                    print(f"[SUCCESS] {len(server_tools)} herramientas cargadas de {url}")

                    # 4. Envolvemos cada herramienta con el closure vinculado a ESTA sesión
                    for tool in server_tools:
                        def create_async_tool_handler(tool_name, bound_session):
                            async def tool_executor(*args, config=None, **kwargs):
                                try:
                                    mcp_args = {}
                                    for key, value in kwargs.items():
                                        if key in ["run_manager", "callbacks", "tags", "metadata", "config"]:
                                            continue
                                        try:
                                            json.dumps(value)
                                            mcp_args[key] = value
                                        except TypeError:
                                            pass
                                    result = await bound_session.call_tool(tool_name, arguments=mcp_args)
                                    text_content = str(result)
                                    return text_content, result
                                except Exception as e:
                                    print(f"[ERROR] Excepción en herramienta {tool_name}: {e}")
                                    return f"Error: Falló la ejecución de la herramienta '{tool_name}'. Detalle: {str(e)}", None
                            return tool_executor

                        tool._arun = create_async_tool_handler(tool.name, session)
                        all_tools.append(tool)

                except Exception as e:
                    # Si un MCP está caído, no rompemos el resto
                    print(f"[WARNING] Fallo al conectar con MCP {url}: {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue

            if not all_tools:
                raise RuntimeError("No se pudo cargar ninguna herramienta de los servidores MCP.")

            print(f"[INFO] Agente compilado con un total de {len(all_tools)} herramientas combinadas.")

            # 5. Compilar el agente de LangGraph con las herramientas unificadas
            agent_executor = create_agent(all_tools)
            print("[INFO] Agente compilado dinámicamente. Iniciando ejecución del grafo...")

            # 6. Transmitir los tokens en formato SSE
            in_tool_call = False
            in_think = False

            async for event in agent_executor.astream_events(inputs, config=config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    content = chunk.content
                    if content:
                        # Ocultar tags de pensamiento
                        if "<think>" in content:
                            in_think = True
                        if in_think:
                            if "</think>" in content:
                                in_think = False
                                content = content.split("</think>")[-1]
                            else:
                                continue

                        # Ocultar tags de llamadas a herramientas
                        if "<tool_call>" in content:
                            in_tool_call = True

                        if in_tool_call:
                            if "</tool_call>" in content:
                                in_tool_call = False
                                content = content.split("</tool_call>")[-1]
                            else:
                                continue

                        if content and not in_tool_call and not in_think:
                            safe_content = json.dumps(content)
                            yield f"data: {safe_content}\n\n"
                            await asyncio.sleep(0.01)

            print("[INFO] Flujo del agente finalizado. Cerrando conexiones MCP...")
                
    except ExceptionGroup as eg:
        # Esto atrapará los errores dentro del TaskGroup en Python 3.11+
        print("[ERROR CRÍTICO] Excepciones múltiples en el TaskGroup de MCP:")
        for exc in eg.exceptions:
            print(f" -> Sub-excepción: {type(exc).__name__}: {exc}")
            # Esto imprimirá la línea exacta de la librería que está fallando
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        
        yield format_sse("[ERROR: Fallo de conexión SSE con MCP. Revisa los logs del orquestador]")

    except Exception as e:
        print(f"[ERROR] Fallo general en el flujo del cliente MCP / LangGraph: {e}")
        traceback.print_exc(file=sys.stdout)
        yield format_sse(f"[ERROR: Fallo al conectar o procesar con el servidor MCP: {str(e)}]")


# =====================================================================
# 5. ENDPOINT DE WEBHOOK POST
# =====================================================================

@app.post("/webhook")
async def webhook_endpoint(request_data: WebhookRequest):
    """
    Webhook principal. Recibe peticiones HTTP POST con 'url' y 'message'.
    """
    try:
        url_timeout = float(os.getenv("URL_VALIDATION_TIMEOUT", "5"))
    except ValueError:
        url_timeout = 5.0

    print(f"[INFO] Petición de Webhook recibida.")
    print(f"   > URL a validar: {request_data.url}")
    print(f"   > Mensaje: {request_data.message[:50]}...")
    print(f"   > Session ID: {request_data.session_id}")

    # Validar URL asíncronamente
    is_url_valid = await validate_url_async(request_data.url, url_timeout)
    if not is_url_valid:
        print(f"[WARNING] Validación de URL fallida para {request_data.url}. Abortando petición.")
        raise HTTPException(
            status_code=400,
            detail=f"La URL proporcionada '{request_data.url}' no respondió con un estado HTTP 200 OK. Flujo cancelado."
        )

    print("[SUCCESS] URL validada. Iniciando streaming SSE del agente...")
    
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
    Ruta para la verificación del estado de salud de la aplicación.
    """
    return {
        "status": "ok",
        "mcp_servers": MCP_SERVERS
    }


# =====================================================================
# 7. EJECUCIÓN LOCAL
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    try:
        port = int(os.getenv("AGENT_PORT", "8080"))
    except ValueError:
        port = 8080
        
    print(f"[INFO] Iniciando Uvicorn en el puerto {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
