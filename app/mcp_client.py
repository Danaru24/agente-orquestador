import os
from typing import List
from langchain_mcp_adapters.tools import load_mcp_tools

class ZabbixMCPClient:
    """
    Cliente encargado de la introspección y conexión con el servidor FastMCP.
    """
    def __init__(self):
        # El valor se extrae de la variable de entorno ZABBIX_MCP_URL
        # <---- CONFIGURAR EN CONFIGMAP ----->
        self.mcp_url = os.getenv("ZABBIX_MCP_URL", "http://localhost:8000/sse")

    async def fetch_tools(self) -> List:
        """
        Realiza la conexión SSE y recupera las definiciones de herramientas.
        """
        try:
            # load_mcp_tools se encarga de convertir el protocolo MCP a Tools de LangChain
            tools = await load_mcp_tools(self.mcp_url)
            print(f"✅ Conexión establecida con el MCP en: {self.mcp_url}")
            print(f"📦 Total de herramientas Zabbix detectadas: {len(tools)}")
            return tools
        except Exception as e:
            print(f"❌ Error crítico de comunicación con MCP: {e}")
            return []