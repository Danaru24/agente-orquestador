import os
import re
import json
import uuid
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, trim_messages
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
        # Recortar el historial para conservar solo los últimos 5 mensajes de la conversación
        trimmed_history = trim_messages(
            state["messages"],
            max_tokens=5,
            strategy="last",
            token_counter=len,
        )

        # Filtrado de historial: pasar al LLM solo HumanMessage y AIMessage finales.
        # ToolMessages e AIMessages intermedios (con tool_calls) se descartan del contexto
        # activo igual que se descartan del checkpointer al finalizar el turno.
        clean_history = [
            msg for msg in trimmed_history
            if not (type(msg).__name__ == "ToolMessage" or getattr(msg, "type", "") == "tool")
            and not (
                (type(msg).__name__ == "AIMessage" or getattr(msg, "type", "") == "ai")
                and getattr(msg, "tool_calls", None)
            )
        ]

        system_prompt = SystemMessage(
            content=(
                "Eres un asistente técnico especializado que opera EXCLUSIVAMENTE como un orquestador de herramientas a través de los servidores MCP conectados. \n"
                "Tus directrices de comportamiento son absolutas:\n"
                "- No inventes información, estados, ni respuestas si no han sido recuperados explícitamente por una herramienta MCP.\n"
                "- NO actúes como una guía de comandos. Está estrictamente prohibido devolver comandos de Kubernetes o instrucciones de terminal en formato de texto a menos que el usuario te lo pida explícitamente. Tu deber es EJECUTAR las herramientas del MCP para interactuar con el clúster, no enseñarle comandos al usuario.\n"
                "- Si no cuentas con una herramienta MCP para resolver la petición, indícalo de manera directa y concreta.\n"
                "- Mantén siempre una estructura visual limpia, fácil de leer y con información sumamente concreta (usa viñetas o tablas cortas si es necesario), evitando rodeos o texto innecesario.\n\n"
                "FORMATO DE SALIDA: Está estrictamente prohibido usar bloques de código Markdown (```) para listar información de infraestructura, estado de recursos o inventarios. Utiliza únicamente texto plano con viñetas (-) o tablas simples. Los bloques de código SOLO deben usarse si el usuario pide explícitamente un script, un archivo YAML o código fuente.\n\n"
                "RAZONAMIENTO DE DOMINIO: Tienes acceso a dos entornos distintos: Kubernetes/OpenShift y Zabbix. \n"
                "- Términos como 'Proyecto', 'Namespace', 'Pod', 'Deployment', 'Log' o 'Cluster' pertenecen a KUBERNETES. \n"
                "- Términos como 'Host', 'Grupo de hosts', 'Item', 'Trigger' o 'Métrica de red' pertenecen a ZABBIX.\n"
                "REGLA DE ORO: Si el usuario usa un término ambiguo (como 'inventario') o no estás 100% seguro de a qué entorno se refiere, DEBES PREGUNTAR y pedir aclaración antes de ejecutar cualquier herramienta. No asumas el entorno.\n\n"
                "PREVENCIÓN DE SOBRECARGA: Tienes ESTRICTAMENTE PROHIBIDO ejecutar herramientas de listado (como pods_list, resources_list, etc.) sin proporcionar parámetros de filtrado. Nunca intentes listar los recursos de todo el clúster globalmente enviando argumentos vacíos ({}). Si buscas algo y no lo encuentras en el namespace indicado, no hagas una búsqueda a ciegas; detente y pídele al usuario que verifique el namespace.\n\n"
                "GESTIÓN DE ERRORES Y HERRAMIENTAS: \n"
                "- Si buscas un Deployment, Service o Ingress, NO inventes herramientas como 'deployments_list'. Revisa la lista de herramientas disponibles y usa las genéricas como 'resources_list' o busca los pods directamente con 'pods_list'.\n"
                "- Si una herramienta te devuelve un error indicando que no es válida, TIENES PROHIBIDO rendirte y mostrar código YAML o plantillas genéricas. Debes corregir tu llamada utilizando una de las herramientas sugeridas en el mensaje de error."
            )
        )
        
        messages = [system_prompt] + clean_history

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
