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
from mcp.client.streamable_http import streamable_http_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import RemoveMessage

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
# Cada URL debe apuntar al endpoint EXACTO de streaming de cada servidor (sin modificaciones)
MCP_SERVERS = [
    url.strip() for url in os.getenv(
        "MCP_SERVERS",
        "https://kubernetes-mcp-server-infra-ai.apps.ocp.zz987.sandbox2813.opentlc.com/sse,https://zabbix-mcp-server-agente-command.apps.ocp.zz987.sandbox2813.opentlc.com/mcp"
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
# 3.5. LIMPIEZA DE TABLAS Y COLUMNAS INNECESARIAS (PRE-PROCESAMIENTO)
# =====================================================================

def clean_mcp_output(text: str) -> str:
    """
    Intercepta el texto devuelto por las herramientas del MCP.
    Elimina columnas innecesarias de salidas tabulares y hashes largos.
    """
    if not text or not isinstance(text, str):
        return text

    lines = text.split("\n")
    header_idx = -1
    for idx, line in enumerate(lines):
        # Buscamos la fila de cabecera típica de kubectl
        if any(keyword in line for keyword in ["NAME", "READY", "STATUS", "RESTARTS", "AGE", "NOMINATED"]):
            header_idx = idx
            break

    if header_idx == -1:
        # No se encontró tabla típica, se devuelve el texto con limpieza de hashes
        return re.sub(r'(?<!app=)(?<!pod-template-hash=)\b[a-f0-9]{10,}\b', '[hash]', text)

    header_line = lines[header_idx]
    # Dividir las columnas basándonos en 2 o más espacios consecutivos
    columns = re.split(r'\s{2,}', header_line.strip())
    if len(columns) <= 1:
        return re.sub(r'(?<!app=)(?<!pod-template-hash=)\b[a-f0-9]{10,}\b', '[hash]', text)

    # Columnas que deseamos excluir por completo
    cols_to_exclude = {"READINESS GATES", "NOMINATED NODE", "AGE"}
    col_ranges = []
    current_pos = 0
    for col in columns:
        start = header_line.find(col, current_pos)
        end = start + len(col)
        current_pos = end
        col_ranges.append((col, start, end))

    keep_indices = []
    for i, (col, start, end) in enumerate(col_ranges):
        if col.upper() not in cols_to_exclude:
            keep_indices.append(i)

    # Calcular los cortes (slices) de cada columna a conservar
    slices = []
    for i in keep_indices:
        start = col_ranges[i][1]
        if i + 1 < len(col_ranges):
            end = col_ranges[i+1][1]
        else:
            end = None
        slices.append((start, end))

    cleaned_lines = []
    for idx, line in enumerate(lines):
        if idx < header_idx:
            # Preservar líneas previas (como warnings o descripciones)
            cleaned_lines.append(line)
            continue
        if not line.strip():
            cleaned_lines.append(line)
            continue
        row_parts = []
        for start, end in slices:
            part = line[start:end].rstrip() if end is not None else line[start:].rstrip()
            row_parts.append(part)
        cleaned_lines.append("    ".join(row_parts))

    cleaned_text = "\n".join(cleaned_lines)
    # Reemplazar hashes largos de hexadecimales que no pertenezcan a app= o pod-template-hash=
    cleaned_text = re.sub(r'(?<!app=)(?<!pod-template-hash=)\b[a-f0-9]{10,}\b', '[hash]', cleaned_text)
    return cleaned_text


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

    try:
        async with AsyncExitStack() as stack:
            all_tools = []

            # Conectamos con cada servidor de la lista usando la URL tal cual
            for url in MCP_SERVERS:
                try:
                    if url.endswith("/sse"):
                        # 1. Abrimos conexión SSE con la URL exacta (sin modificaciones)
                        streams = await stack.enter_async_context(sse_client(url=url, timeout=None))
                        read_stream, write_stream = streams
                    else:
                        # 1. Abrimos conexión Streamable HTTP con la URL exacta (sin modificaciones)
                        streams = await stack.enter_async_context(streamable_http_client(url=url))
                        read_stream, write_stream, _ = streams

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
                                    # Interceptor de argumentos vacíos para pods_list
                                    if tool_name == "pods_list" and not mcp_args.get("namespace"):
                                        return "Error: No se permite listar pods sin especificar un namespace.", None

                                    result = await bound_session.call_tool(tool_name, arguments=mcp_args)
                                    text_content = str(result)

                                    # Limpieza de tablas y columnas innecesarias (Pre-procesamiento)
                                    text_content = clean_mcp_output(text_content)

                                    # Truncamiento de salidas masivas (Output Clipping)
                                    if len(text_content) > 2000:
                                        text_content = (
                                            text_content[:2000] +
                                            "\n\n[SYSTEM] La salida anterior ha sido truncada porque superó el límite de contexto. "
                                            "Refina la búsqueda usando argumentos de filtrado (namespace, labelSelector, etc.) "
                                            "antes de continuar. NO repitas esta instrucción al usuario."
                                        )

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

            try:
                # Log de tamaño de payload (pre-ejecución)
                tools_payload_desc = str([{ "name": t.name, "description": t.description, "args": getattr(t, "args", {}) } for t in all_tools])
                approx_system_prompt_len = 1000  # Estimación del System Prompt estricto
                total_approx_chars = len(tools_payload_desc) + approx_system_prompt_len
                print(f"[DEBUG] [PAYLOAD_LOG] Cantidad exacta de herramientas inyectadas: {len(all_tools)}")
                print(f"[DEBUG] [PAYLOAD_LOG] Longitud aproximada del payload de herramientas: {len(tools_payload_desc)} caracteres")
                print(f"[DEBUG] [PAYLOAD_LOG] Longitud aproximada de (herramientas + System Prompt): {total_approx_chars} caracteres")

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

                # Al finalizar el flujo, limpiamos el historial persistente de la sesión (checkpointer)
                # para conservar únicamente los HumanMessages y los AIMessages finales (sin tool_calls)
                try:
                    state = await agent_executor.aget_state(config)
                    all_messages = state.values.get("messages", [])
                    
                    messages_to_remove = []
                    for msg in all_messages:
                        if type(msg).__name__ == "ToolMessage" or getattr(msg, "type", "") == "tool":
                            messages_to_remove.append(RemoveMessage(id=msg.id))
                        elif type(msg).__name__ == "AIMessage" or getattr(msg, "type", "") == "ai":
                            if msg.tool_calls:
                                messages_to_remove.append(RemoveMessage(id=msg.id))
                    
                    if messages_to_remove:
                        print(f"[INFO] Compresión de memoria: eliminando {len(messages_to_remove)} mensajes intermedios del checkpointer...")
                        await agent_executor.aupdate_state(config, {"messages": messages_to_remove})
                except Exception as clean_err:
                    print(f"[WARNING] No se pudo limpiar el historial persistente: {clean_err}")

            except Exception as e:
                err_msg = str(e)
                exc_type = type(e).__name__
                print(f"[ERROR CRÍTICO] [INFERENCE_ENGINE_FAIL] Excepción capturada en la ejecución del agente ({exc_type}): {err_msg}")
                traceback.print_exc(file=sys.stdout)
                
                # Cierre limpio de sesiones y conexiones MCP para evitar fugas de memoria
                print("[INFO] Cerrando de forma limpia las conexiones MCP...")
                await stack.aclose()
                
                # Degradación elegante con mensaje amigable al usuario
                friendly_error = "Error interno en el motor de inferencia. Es posible que la consulta o las herramientas hayan excedido la capacidad actual del modelo."
                yield f"data: {json.dumps(friendly_error)}\n\n"
                
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
