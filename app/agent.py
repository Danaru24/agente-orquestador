import os
from typing import Annotated, TypedDict, List

# Librerías de LangChain & LangGraph
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

# Importamos tu cliente del Paso 1
from mcp_client import ZabbixMCPClient

# --- DEFINICIÓN DEL ESTADO ---
class AgentState(TypedDict):
    """
    Representa la memoria de corto plazo del agente. 
    'add_messages' permite que el historial se concatene automáticamente.
    """
    messages: Annotated[List[BaseMessage], add_messages]

async def create_agent():
    # ---- CONFIGURACIÓN DESDE VARIABLES DE ENTORNO (CONFIGMAP) ----
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2")
    SYSTEM_PROMPT_TEXT = os.getenv(
        "SYSTEM_PROMPT", 
        "Eres un asistente técnico de SRE. Ayudas a monitorear infraestructura con Zabbix."
    )

    # 1. Cargamos las herramientas desde el MCP de Zabbix
    zabbix_mcp = ZabbixMCPClient()
    zabbix_tools = await zabbix_mcp.fetch_tools()

    # 2. Inicializamos el modelo Llama 3.2
    # Temperature 0 es vital para evitar alucinaciones en datos de infraestructura
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_URL,
        temperature=0
    ).bind_tools(zabbix_tools) # Le damos acceso a las herramientas de Zabbix

    # 3. Definimos los NODOS del Grafo
    
    def call_model(state: AgentState):
        """Nodo que procesa la lógica del lenguaje."""
        sys_message = SystemMessage(content=SYSTEM_PROMPT_TEXT)
        # Invocamos al modelo con el prompt de sistema y el historial de mensajes
        response = llm.invoke([sys_message] + state["messages"])
        return {"messages": [response]}

    # Nodo pre-construido que ejecuta las herramientas del MCP
    tool_node = ToolNode(zabbix_tools)

    # 4. Construcción de la LÓGICA (El Grafo)
    workflow = StateGraph(AgentState)

    # Añadimos los puntos de procesamiento
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Establecemos el punto de entrada
    workflow.set_entry_point("agent")

    # --- FLUJO CONDICIONAL ---
    # Después de 'agent', revisamos si el modelo pidió usar una herramienta
    workflow.add_conditional_edges(
        "agent",
        tools_condition, # Si hay tool_calls -> 'tools', si no -> END
    )

    # Después de ejecutar la herramienta en Zabbix, regresamos al agente 
    # para que explique el resultado en lenguaje humano
    workflow.add_edge("tools", "agent")

    # Compilamos el grafo
    return workflow.compile()