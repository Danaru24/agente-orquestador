import os
import re
import json
import uuid
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

# Instanciamos el checkpointer en memoria de forma global para mantener el historial entre peticiones
memory_checkpointer = MemorySaver()

# =====================================================================
# 1. DEFINICIÓN DEL ESTADO DEL AGENTE (AgentState)
# =====================================================================

class AgentState(TypedDict):
    """
    Define el estado compartido a lo largo de la ejecución del grafo.
    
    Esta clase hereda de TypedDict y contiene la clave 'messages'.
    El decorador 'Annotated' junto con la función 'add_messages' de LangGraph
    indica que las actualizaciones a la lista de mensajes no sobreescribirán
    el estado anterior, sino que concatenarán (anexarán) los nuevos mensajes
    al historial de forma automática.
    """
    messages: Annotated[List[BaseMessage], add_messages]


# =====================================================================
# 2. CONSTRUCCIÓN DEL FLUJO (Grafo) Y VINCULACIÓN DE HERRAMIENTAS
# =====================================================================

def create_agent(tools: list) -> CompiledStateGraph:
    """
    Inicializa el LLM local, asocia las herramientas del servidor MCP remoto
    recibidas por parámetro, y compila el flujo (grafo) de LangGraph.
    
    Args:
        tools (list): Lista de herramientas de LangChain mapeadas desde el MCP
                      que serán vinculadas al LLM.
                      
    Returns:
        CompiledStateGraph: El grafo de LangGraph compilado y listo para invocar.
    """
    # Extraemos las configuraciones desde variables de entorno
    local_model_url = os.getenv("LOCAL_MODEL_URL", "http://localhost:8000/v1")
    local_model_name = os.getenv("LOCAL_MODEL_NAME", "local-model")
    simulated_api_key = os.getenv("SIMULATED_API_KEY", "mock-api-key-12345")
    
    try:
        llm_timeout = float(os.getenv("LLM_TIMEOUT", "30"))
    except ValueError:
        llm_timeout = 30.0

    print("--- [AGENTE] CONFIGURACIÓN DEL MODELO LOCAL ---")
    print(f"URL Base: {local_model_url}")
    print(f"Nombre del Modelo: {local_model_name}")
    print(f"Timeout del LLM: {llm_timeout}s")
    print(f"Herramientas vinculadas: {len(tools)}")
    print("-----------------------------------------------")

    # Inicializamos el cliente de ChatOpenAI apuntando al modelo local
    llm = ChatOpenAI(
        base_url=local_model_url,
        model=local_model_name,
        api_key=simulated_api_key,
        timeout=llm_timeout,
        temperature=0.3
    )

    # Vinculamos las herramientas MCP mapeadas al modelo de lenguaje
    # Si la lista de herramientas está vacía, no se bindean herramientas.
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    # =====================================================================
    # Nodos del Grafo
    # =====================================================================

    def call_model(state: AgentState) -> dict:
        """
        Nodo del agente que realiza la invocación al modelo de lenguaje.
        
        Toma el historial actual de mensajes del AgentState, le antepone un 
        mensaje del sistema (SystemMessage) para fijar el comportamiento 
        del asistente, e invoca al LLM.
        
        Args:
            state (AgentState): El estado actual con el historial de mensajes.
            
        Returns:
            dict: Una actualización para el estado que contiene el mensaje generado.
        """
        # ── Ventana deslizante de memoria (scope correcto) ─────────────────
        # REGLA FUNDAMENTAL:
        #   - Los mensajes del TURNO ACTUAL (desde el último HumanMessage hasta
        #     el final) se pasan siempre INTACTOS al LLM. El modelo necesita ver
        #     los ToolMessages y sus propios tool_calls para no entrar en bucle.
        #   - Los mensajes de TURNOS ANTERIORES se filtran (solo Human ↔ AI
        #     final) y se limitan a los últimos 5 turnos válidos para reducir
        #     el tamaño del contexto.
        all_msgs = state["messages"]

        # Localizar el índice del último HumanMessage (inicio del turno actual)
        current_turn_start = 0
        for i in range(len(all_msgs) - 1, -1, -1):
            if type(all_msgs[i]).__name__ == "HumanMessage" or getattr(all_msgs[i], "type", "") == "human":
                current_turn_start = i
                break

        # Historial anterior (turnos pasados): se filtra y se limita a 5 turnos
        past_msgs = all_msgs[:current_turn_start]
        past_clean = [
            msg for msg in past_msgs
            if not (type(msg).__name__ == "ToolMessage" or getattr(msg, "type", "") == "tool")
            and not (
                (type(msg).__name__ == "AIMessage" or getattr(msg, "type", "") == "ai")
                and getattr(msg, "tool_calls", None)
            )
        ][-5:]  # ventana deslizante de 5 turnos Human↔AI

        # Turno actual (desde el último HumanMessage): se pasa íntegro
        current_turn_msgs = all_msgs[current_turn_start:]

        # Contexto final que recibe el LLM en esta invocación
        active_context = past_clean + current_turn_msgs

        system_prompt = SystemMessage(
            content=(
                "Eres un orquestador técnico de infraestructura con acceso a Kubernetes/OpenShift y Zabbix vía MCP.\n"
                "REGLAS CRÍTICAS:\n"
                "1. NUNCA INVENTES: Usa las herramientas para responder. Si no tienes la herramienta adecuada, indícalo.\n"
                "2. FILTRADO OBLIGATORIO: Prohibido ejecutar listados globales (ej. pods_list sin argumentos). "
                "Si necesitas buscar recursos, pide siempre el namespace o host primero.\n"
                "3. FORMATO LIMPIO: Usa texto plano y viñetas. NO uses bloques de código Markdown (```) "
                "a menos que el usuario pida explícitamente un script o YAML.\n"
                "4. CORRECCIÓN AUTOMÁTICA: Si una herramienta falla, lee el error y usa la herramienta correcta. "
                "No te disculpes ofreciendo plantillas YAML genéricas.\n"
                "5. ANTI-BUCLES: Tienes estrictamente prohibido llamar a la misma herramienta dos veces seguidas "
                "para la misma petición. Una vez que llames a una herramienta y recibas su resultado, DEBES analizar "
                "esa información y responder inmediatamente al usuario. NO vuelvas a intentar ejecutar herramientas."
            )
        )
        
        messages = [system_prompt] + active_context

        # Función local para escribir en llm_context_debug.log (sobreescritura).
        # SOLO escribe: System Prompt · marcador de herramientas · historial limpio · última pregunta.
        # Los esquemas JSON de herramientas se omiten completamente para mantener el archivo legible.
        def log_llm_context(final_messages: list, num_tools: int):
            try:
                log_path = "llm_context_debug.log"
                SEP = "=====================================================================\n"

                # 1. System Prompt
                system_content = next(
                    (m.content for m in final_messages if isinstance(m, SystemMessage)),
                    "No system prompt found"
                )

                # 2. Historial limpio (solo HumanMessage + AIMessage final sin tool_calls)
                clean_log_history = [
                    m for m in final_messages
                    if not isinstance(m, SystemMessage)
                    and not (type(m).__name__ == "ToolMessage" or getattr(m, "type", "") == "tool")
                    and not (
                        (type(m).__name__ == "AIMessage" or getattr(m, "type", "") == "ai")
                        and getattr(m, "tool_calls", None)
                    )
                ]

                # 3. Última pregunta del usuario
                user_msgs = [
                    m for m in final_messages
                    if type(m).__name__ == "HumanMessage" or getattr(m, "type", "") == "human"
                ]
                last_question = user_msgs[-1].content if user_msgs else "No user question found in this context."

                with open(log_path, "w", encoding="utf-8") as f:
                    # --- SYSTEM PROMPT ---
                    f.write(SEP + " SYSTEM PROMPT\n" + SEP)
                    f.write(system_content + "\n\n")

                    # --- AVAILABLE TOOLS (abreviado) ---
                    f.write(SEP + " AVAILABLE TOOLS\n" + SEP)
                    f.write(f"... [LISTA DE HERRAMIENTAS MCP OMITIDAS] ... ({num_tools} herramientas activas)\n\n")

                    # --- TRIMMED CONVERSATION HISTORY ---
                    f.write(SEP + " TRIMMED CONVERSATION HISTORY (LAST 5 MESSAGES — solo Human & AI final)\n" + SEP)
                    if clean_log_history:
                        for idx, msg in enumerate(clean_log_history):
                            msg_type = type(msg).__name__
                            f.write(f"[{msg_type}] ({idx + 1}/{len(clean_log_history)}):\n")
                            f.write(f"{msg.content}\n")
                            f.write("-" * 80 + "\n")
                    else:
                        f.write("(historial vacío)\n")
                    f.write("\n")

                    # --- LAST USER QUESTION ---
                    f.write(SEP + " LAST USER QUESTION\n" + SEP)
                    f.write(last_question + "\n")

            except Exception as e:
                print(f"[ERROR] Fallo al escribir en llm_context_debug.log: {e}")

        # Ejecutamos el log del contexto antes de invocar al LLM
        log_llm_context(messages, len(tools))

        response = llm_with_tools.invoke(messages)
        
        # Parseo manual para modelos locales que devuelven tool calls (formato XML o JSON crudo al inicio)
        content = response.content
        tool_calls = []
        if isinstance(content, str):
            # Caso 1: Formato XML (<tool_call>...</tool_call>)
            if "<tool_call>" in content:
                pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
                for m in pattern.finditer(content):
                    json_str = m.group(1).strip()
                    try:
                        tool_data = json.loads(json_str)
                        tool_name = tool_data.get("name")
                        tool_args = tool_data.get("args") or tool_data.get("arguments") or {}
                        tool_calls.append({
                            "name": tool_name,
                            "args": tool_args,
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "tool_call"
                        })
                    except Exception as parse_err:
                        print(f"[ERROR] Error al parsear tool_call XML JSON: {parse_err}")
            # Caso 2: Formato JSON crudo al inicio del mensaje
            elif content.strip().startswith("{"):
                stripped_content = content.strip()
                # Buscamos la primera estructura JSON de tool call
                json_match = re.match(r'^\s*(\{"name":.*?"arguments":\s*\{.*?\}\})', stripped_content, re.DOTALL)
                if not json_match:
                    json_match = re.match(r'^\s*(\{"name":.*?"args":\s*\{.*?\}\})', stripped_content, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        tool_data = json.loads(json_str)
                        tool_name = tool_data.get("name")
                        tool_args = tool_data.get("args") or tool_data.get("arguments") or {}
                        tool_calls.append({
                            "name": tool_name,
                            "args": tool_args,
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "tool_call"
                        })
                    except Exception as parse_err:
                        print(f"[ERROR] Error al parsear tool_call JSON crudo: {parse_err}")

            if tool_calls:
                print(f"[DEBUG] Se detectaron y parsearon {len(tool_calls)} tool calls de forma manual.")
                new_response = AIMessage(
                    content="",
                    tool_calls=tool_calls,
                    id=response.id
                )
                response = new_response

        return {"messages": [response]}

    # =====================================================================
    # Construcción del Grafo
    # =====================================================================
    
    workflow = StateGraph(AgentState)

    # Registramos el nodo principal del modelo
    workflow.add_node("agent", call_model)

    # Solo agregamos el nodo de herramientas si existen herramientas mapeadas
    if tools:
        workflow.add_node("tools", ToolNode(tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")
    else:
        # Si no hay herramientas, el flujo simplemente termina en el agente
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)

    # Compilamos el grafo con el checkpointer global
    return workflow.compile(checkpointer=memory_checkpointer)
