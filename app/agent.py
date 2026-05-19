import os
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI # [Cambiado de langchain_ollama]
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from mcp_client import ZabbixMCPClient

# --- DEFINICIÓN DEL ESTADO ---
class AgentState(TypedDict):
    """
    Representa la memoria de corto plazo del agente. 
    'add_messages' permite que el historial se concatene automáticamente.
    """
    messages: Annotated[List[BaseMessage], add_messages]

async def create_agent():
    # ---- NUEVAS VARIABLES PARA LLAMA.CPP ----
    LLAMACPP_URL = os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080/v1")
    MODEL_NAME = os.getenv("LLM_MODEL", "local-model")
    SYSTEM_PROMPT_TEXT = os.getenv("SYSTEM_PROMPT", "Eres un asistente técnico de SRE.")

    zabbix_mcp = ZabbixMCPClient()
    zabbix_tools = await zabbix_mcp.fetch_tools()

    # 2. Inicializamos el modelo usando la interfaz de OpenAI
    # El servidor de llama.cpp emula esta API
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=LLAMACPP_URL,
        api_key="no-needed", # llama.cpp no suele requerir API Key local
        temperature=0
    ).bind_tools(zabbix_tools)
    
    def call_model(state: AgentState):
        sys_message = SystemMessage(content=SYSTEM_PROMPT_TEXT)
        response = llm.invoke([sys_message] + state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(zabbix_tools)
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()
