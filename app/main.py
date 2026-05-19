import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Importamos el cerebro del agente (Paso 2)
from agent import create_agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sovereign Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, podrías restringirlo a la URL de tu Open WebUI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global para el grafo compilado
agent_executor = None

@app.on_event("startup")
async def startup_event():
    """Se ejecuta al arrancar el contenedor en OpenShift."""
    global agent_executor
    print("🚀 Inicializando Agente y conectando con MCP y llama.cpp...")
    # --- PRINTS DE VERIFICACIÓN DE ENTORNO ---
    print("--- VERIFICACIÓN DE VARIABLES DE ENTORNO ---")
    print(f"PORT: {os.getenv('AGENT_PORT')}")
    print(f"LOG_LEVEL: {os.getenv('LOG_LEVEL')}")
    print(f"SYSTEM_PROMPT: {os.getenv('SYSTEM_PROMPT')[:50]}...") # Imprimimos solo el inicio
    print("-------------------------------------------")
    agent_executor = await create_agent()
    print("✅ Agente listo para recibir consultas.")

# --- SCHEMAS PARA COMPATIBILIDAD CON OPENAI/OPEN WEBUI ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False

# --- ENDPOINT PRINCIPAL ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint que emula la API de OpenAI para ser consumida por Open WebUI.
    """
    if not agent_executor:
        raise HTTPException(status_code=503, detail="Agente no inicializado")

    try:
        # 1. Convertimos los mensajes de la petición al formato de LangChain
        # LangGraph espera una lista de mensajes (HumanMessage, AIMessage, etc.)
        inputs = {"messages": [("user", msg.content) for msg in request.messages]}

        # 2. Ejecutamos el Grafo (Cerebro)
        # Usamos config para separar hilos de conversación si fuera necesario
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = await agent_executor.ainvoke(inputs, config=config)

        # 3. Extraemos la última respuesta del modelo
        final_message = result["messages"][-1].content

        # 4. Formateamos la respuesta como la espera OpenAI/Open WebUI
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": 1677652288, # Timestamp ficticio
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_message
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    except Exception as e:
        print(f"❌ Error procesando la petición: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Ruta para los Liveness/Readiness probes de OpenShift."""
    return {"status": "ok", "agent_loaded": agent_executor is not None}

if __name__ == "__main__":
    import uvicorn
    # Leemos el puerto del entorno (estándar en OpenShift es 8080)
    port = int(os.getenv("AGENT_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
