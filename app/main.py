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

# URL base del servidor MCP remoto para Kubernetes usando transporte SSE
K8S_MCP_URL = os.getenv(
    "K8S_MCP_URL",
    "https://kubernetes-mcp-server-infra-ai.apps.ocp.zz987.sandbox2813.opentlc.com/mcp"
)

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
    print(f"[STARTUP] Servidor MCP configurado en: {K8S_MCP_URL}")
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

    print(f"[INFO] Iniciando conexión SSE con el servidor MCP en: {K8S_MCP_URL}")

    # Para la conexión SSE, el cliente de Python requiere apuntar al endpoint /sse.
    # Si la URL configurada termina en /mcp, la redirigimos dinámicamente a /sse.
    sse_url = K8S_MCP_URL
    if sse_url.endswith("/mcp"):
        sse_url = sse_url[:-4] + "/sse"
    elif not sse_url.endswith("/sse"):
        sse_url = sse_url.rstrip("/") + "/sse"

    try:
        # 1. Establecer conexión SSE mediante sse_client con timeout elevado para evitar caídas
        async with sse_client(url=sse_url, timeout=None) as streams:
            read_stream, write_stream = streams
            
            # 2. Inicializar la sesión del protocolo MCP
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("[SUCCESS] Sesión MCP inicializada correctamente.")
                
                # 3. Descargar y mapear herramientas MCP a LangChain de forma dinámica
                tools = await load_mcp_tools(session)
                print(f"[SUCCESS] {len(tools)} herramientas cargadas desde el servidor MCP.")

                # Envolver ejecución de herramientas en un try-except seguro para evitar que errores o timeouts crasheen la conexión SSE
                wrapped_tools = []
                for tool in tools:
                    def create_async_tool_handler(tool_name):
                        async def tool_executor(*args, config=None, **kwargs):
                            try:
                                # 1. Escudo de serialización a prueba de LangChain
                                mcp_args = {}
                                for key, value in kwargs.items():
                                    if key in ["run_manager", "callbacks", "tags", "metadata", "config"]:
                                        continue
                                    try:
                                        json.dumps(value)
                                        mcp_args[key] = value
                                    except TypeError:
                                        pass
                                
                                # 2. Ejecutamos la llamada al servidor MCP con los argumentos limpios
                                result = await session.call_tool(tool_name, arguments=mcp_args)
                                
                                # 3. Retornamos la tupla esperada por LangGraph ('content_and_artifact')
                                text_content = str(result)
                                return text_content, result
                            except Exception as e:
                                print(f"[ERROR] Excepción capturada en ejecución asíncrona de la herramienta {tool_name}: {e}")
                                return f"Error: Falló la ejecución de la herramienta '{tool_name}'. Detalle: {str(e)}", None
                        return tool_executor

                    tool._arun = create_async_tool_handler(tool.name)
                    wrapped_tools.append(tool)
                tools = wrapped_tools
                
                # 4. Compilar el agente de LangGraph con las herramientas cargadas
                agent_executor = create_agent(tools)
                print("[INFO] Agente compilado dinámicamente. Iniciando ejecución del grafo...")
                
                # 5. Transmitir los tokens en formato SSE
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
                                    # Limpiamos la parte del tag si llegó texto útil en el mismo chunk
                                    content = content.split("</tool_call>")[-1]
                                else:
                                    continue # Saltamos el 'yield', suprimiendo este chunk de la interfaz

                            if content and not in_tool_call and not in_think:
                                # Empaquetamos el fragmento en la estructura nativa que espera OpenWebUI
                                chunk_payload = {
                                    "id": "chatcmpl-mcp-agent",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": "agente-orquestador",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": content}
                                        }
                                    ]
                                }
                                # json.dumps protege el texto y el \n viaja seguro dentro de la cadena JSON
                                yield f"data: {json.dumps(chunk_payload)}\n\n"
                                await asyncio.sleep(0.01)

                print("[INFO] Flujo del agente finalizado. Cerrando conexión MCP...")
                
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
        "mcp_url": K8S_MCP_URL
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
