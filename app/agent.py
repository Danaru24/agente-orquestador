import os
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

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
    al historial existente. Esto es crucial para mantener la memoria
    de la conversación entre el humano y el agente inteligente.
    """
    messages: Annotated[List[BaseMessage], add_messages]


# =====================================================================
# 2. HERRAMIENTA SIMULADA (Mock Tool) PARA KUBERNETES
# =====================================================================

@tool
def k8s_pod_status_tool(namespace: str = "default") -> str:
    """
    Simula la obtención del estado de los Pods en un namespace de Kubernetes.
    
    Esta función está preparada para ser reemplazada en el futuro por un
    conector MCP (Model Context Protocol) real de Kubernetes. Permite al agente
    saber si los servicios de infraestructura están respondiendo correctamente.
    
    Args:
        namespace (str): El namespace del clúster que se desea inspeccionar.
                          Por defecto es 'default'.
                          
    Returns:
        str: Un reporte en texto plano detallando el estado de los Pods simulados.
    """
    # Simulamos una respuesta típica de 'kubectl get pods' o una API de monitoreo.
    return (
        f"--- Reporte de Pods Simulados ---\n"
        f"Namespace consultado: {namespace}\n"
        f"Pods en el clúster:\n"
        f"  - api-gateway-7df89bc-abc12    | Ready (1/1) | Status: Running | Restarts: 0 | Age: 4d\n"
        f"  - backend-worker-54c78f-xyz34  | Ready (1/1) | Status: Running | Restarts: 2 | Age: 12h\n"
        f"  - database-postgres-0         | Ready (1/1) | Status: Running | Restarts: 0 | Age: 15d\n"
        f"  - cache-redis-89dfb-qwe99      | Ready (1/1) | Status: Running | Restarts: 1 | Age: 3d\n"
        f"Estado general: SALUDABLE (Todos los Pods principales operativos)"
    )


# =====================================================================
# 3. CONSTRUCCIÓN DEL FLUJO (Grafo) Y LLM
# =====================================================================

def create_agent() -> CompiledStateGraph:
    """
    Inicializa el LLM local, asocia las herramientas y compila el flujo (grafo) de LangGraph.
    
    Este método lee todas las configuraciones necesarias desde variables de entorno
    para facilitar la portabilidad en despliegues sobre clústeres como OpenShift.
    
    Configuraciones de entorno utilizadas:
        - LOCAL_MODEL_URL: URL base del modelo local (vLLM, Ollama, llama.cpp, etc.)
        - LOCAL_MODEL_NAME: Nombre del modelo a utilizar.
        - SIMULATED_API_KEY: Clave de API ficticia requerida por el driver ChatOpenAI.
        - LLM_TIMEOUT: Tiempo de espera máximo para peticiones al LLM.
        
    Returns:
        CompiledStateGraph: El grafo de LangGraph compilado y listo para invocar.
    """
    # Extraemos las configuraciones desde variables de entorno
    # Si no existen, se definen valores por defecto amigables para desarrollo local
    local_model_url = os.getenv("LOCAL_MODEL_URL", "http://localhost:8000/v1")
    local_model_name = os.getenv("LOCAL_MODEL_NAME", "local-model")
    simulated_api_key = os.getenv("SIMULATED_API_KEY", "mock-api-key-12345")
    
    try:
        llm_timeout = float(os.getenv("LLM_TIMEOUT", "30"))
    except ValueError:
        llm_timeout = 30.0

    # Imprimimos logs informativos (ideal para depuración en Kubernetes/OpenShift)
    print("--- [AGENTE] CONFIGURACIÓN DEL MODELO LOCAL ---")
    print(f"URL Base: {local_model_url}")
    print(f"Nombre del Modelo: {local_model_name}")
    print(f"Timeout del LLM: {llm_timeout}s")
    print("-----------------------------------------------")

    # Inicializamos el cliente ChatOpenAI
    # Aunque apunta a un modelo local, usamos ChatOpenAI debido a que la gran mayoría
    # de motores locales de LLM (como Ollama, llama.cpp, vLLM, etc.) exponen
    # una API compatible con el estándar de OpenAI.
    llm = ChatOpenAI(
        base_url=local_model_url,
        model=local_model_name,
        api_key=simulated_api_key,
        timeout=llm_timeout,
        temperature=0.2 # Temperatura baja para respuestas técnicas de SRE más predecibles
    )

    # Definimos la lista de herramientas disponibles para el agente
    tools = [k8s_pod_status_tool]
    
    # Vinculamos las herramientas al modelo de lenguaje
    # Esto permite que el LLM decida de forma autónoma cuándo llamar a una herramienta
    # insertando una sección especial (tool_calls) en sus mensajes de respuesta.
    llm_with_tools = llm.bind_tools(tools)

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
        # Definimos las instrucciones generales del Agente en un SystemMessage
        system_prompt = SystemMessage(
            content=(
                "Eres un agente inteligente experto en administración de sistemas y SRE (Site Reliability Engineering).\n"
                "Tienes acceso a herramientas para monitorear el estado de clústeres de Kubernetes e infraestructura.\n"
                "Usa siempre la herramienta 'k8s_pod_status_tool' si te preguntan por pods o estado del clúster.\n"
                "Tus respuestas deben ser claras, concisas y en idioma español."
            )
        )
        
        # Combinamos el mensaje del sistema con el historial de la conversación
        messages = [system_prompt] + state["messages"]
        
        # Invocamos al LLM configurado con herramientas
        response = llm_with_tools.invoke(messages)
        
        # Retornamos la respuesta dentro de la clave 'messages' para que LangGraph
        # la agregue automáticamente al historial.
        return {"messages": [response]}

    # =====================================================================
    # Construcción del Grafo
    # =====================================================================
    
    # Instanciamos el StateGraph usando nuestro esquema de estado (AgentState)
    workflow = StateGraph(AgentState)

    # Registramos los nodos en el grafo
    # El nodo "agent" procesa el LLM, y el nodo "tools" ejecuta las herramientas
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Establecemos el punto de entrada. Toda ejecución del grafo comenzará en el nodo "agent"
    workflow.set_entry_point("agent")

    # Añadimos enlaces condicionales al nodo "agent"
    # La función preconstruida 'tools_condition' analiza la respuesta del LLM:
    #   - Si el LLM decidió llamar a una herramienta (tool_calls en la respuesta), redirige a "tools".
    #   - Si el LLM generó una respuesta de texto ordinaria, finaliza la ejecución (END).
    workflow.add_conditional_edges(
        "agent",
        tools_condition
    )

    # Añadimos un enlace directo desde "tools" de vuelta a "agent"
    # Después de ejecutar una herramienta, el flujo regresa al modelo para que éste
    # interprete los resultados de la herramienta y formule su respuesta final.
    workflow.add_edge("tools", "agent")

    # Instanciamos un checkpointer en memoria para persistir el historial de mensajes
    checkpointer = MemorySaver()

    # Compilamos el grafo con el checkpointer. Ahora está listo para procesar peticiones
    return workflow.compile(checkpointer=checkpointer)
