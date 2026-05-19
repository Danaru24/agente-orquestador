import os
from typing import List
from langchain_mcp_adapters.tools import load_mcp_tools

import os
import asyncio
from typing import List
from langchain_mcp_adapters.tools import load_mcp_tools

class ZabbixMCPClient:
    def __init__(self):
        self.mcp_url = os.getenv("ZABBIX_MCP_URL", "http://localhost:8000/sse")
        # --- PRINT DE VERIFICACIÓN ---
        print(f"DEBUG: Intentando conectar al MCP usando la URL: {self.mcp_url}")
        self.max_retries = 5
        self.retry_delay = 5  # segundos

    async def fetch_tools(self) -> List:
        """
        Intenta conectar con el servidor MCP con una lógica de reintentos
        para manejar la latencia de red o el arranque de servicios.
        """
        attempt = 10
        while attempt <= self.max_retries:
            try:
                print(f"🔄 Intentando conectar al MCP (Intento {attempt}/{self.max_retries})...")
                # load_mcp_tools realiza la introspección del protocolo
                tools = await load_mcp_tools(self.mcp_url)
                
                if tools:
                    print(f"✅ Conexión exitosa. Herramientas detectadas: {len(tools)}")
                    return tools
                
                print("⚠️ El MCP respondió pero no se encontraron herramientas.")
            
            except Exception as e:
                print(f"❌ Error de comunicación con MCP en {self.mcp_url}: {e}")
            
            if attempt < self.max_retries:
                print(f"⏳ Reintentando en {self.retry_delay} segundos...")
                await asyncio.sleep(self.retry_delay)
            
            attempt += 1

        print("🚨 Se agotaron los reintentos. El Agente iniciará sin herramientas de Zabbix.")
        return []
