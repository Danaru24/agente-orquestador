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
        temperature=0.2
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
        system_prompt = SystemMessage(
            content=(
                "Eres un asistente personal experto con acceso a un servidor MCP de Kubernetes.\n"
                "Tus respuestas deben ser claras, directas y fáciles de leer.\n\n"
                "REGLAS ESTRICTAS:\n"
                "1. Nunca incluyas etiquetas internas como <tool_call> o <think> en tu respuesta final hacia el usuario.\n"
                "2. Cuando listes recursos del clúster (como proyectos, pods, nodos), SIEMPRE utiliza viñetas de Markdown (saltos de línea dobles y guiones '-').\n"
                "3. No amontones la información; usa párrafos estructurados."
            )
        )
        
        messages = [system_prompt] + state["messages"]
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
