"""
Pipe de Integración para Open WebUI - Conexión con el Agente SRE

Este script está estructurado como un 'Pipe' para Open WebUI. Permite canalizar
las conversaciones del chat directamente hacia el Agente SRE de LangGraph.

REGLA DE RED Y COMUNICACIÓN INTERNA (POD A POD):
------------------------------------------------
Este script asume que la comunicación es interna de Pod a Pod dentro del mismo proyecto
(namespace) de OpenShift. Utiliza el servicio DNS de Kubernetes interno para resolver 
la IP del webhook del agente de forma segura, evitando exponer el endpoint del agente
al exterior del clúster de forma innecesaria.
"""

import requests
import json
from typing import Generator, Union, Any
from pydantic import BaseModel, Field

class Pipe:
    class Valves(BaseModel):
        # Valves basadas en Pydantic BaseModel para la configuración desde el panel de Open WebUI
        WEBHOOK_URL: str = Field(
            default="http://agente-webhook:8000/webhook",
            description="URL interna del webhook del Agente FastAPI dentro del clúster."
        )
        TARGET_URL: str = Field(
            default="http://example.com",
            description="URL que será validada asíncronamente por el webhook antes de iniciar el agente."
        )

    def __init__(self):
        # Habilitamos el soporte para streaming en el pipeline de Open WebUI
        self.type = "pipe"
        self.id = "sre_agent_pipeline"
        self.name = "Agente SRE Kubernetes"
        
        # Inicializamos las válvulas con la clase de Pydantic
        self.valves = self.Valves()

    def pipe(self, body: dict, __user__: dict = None) -> Union[str, Generator[str, None, None]]:
        """
        Función principal del Pipe. Intercepta el flujo del chat, envía la petición
        al webhook FastAPI y devuelve un generador para el renderizado en tiempo real.
        
        :param body: Diccionario que representa la petición completa de chat recibida.
        :param __user__: Información del usuario de Open WebUI (opcional).
        :return: String en caso de error o un generador para streaming de tokens.
        """
        # 1. Obtenemos el historial de mensajes de la conversación
        messages = body.get("messages", [])
        if not messages:
            return "Error: No se encontraron mensajes en el cuerpo de la conversación."

        # 2. Interceptamos el último mensaje enviado por el usuario
        last_message = messages[-1].get("content", "")

        # 3. Preparamos el payload y headers para nuestro webhook FastAPI
        payload = {
            "url": self.valves.TARGET_URL,
            "message": last_message,
            "session_id": body.get("chat_id", "openwebui-default-chat")
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            # 4. Enviamos la petición POST al webhook asumiendo red interna (Pod a Pod)
            # Usamos stream=True ya que el endpoint responde con Server-Sent Events (SSE)
            response = requests.post(
                self.valves.WEBHOOK_URL,
                json=payload,
                headers=headers,
                stream=True,
                timeout=45 # Timeout extendido para esperar respuestas del modelo
            )

            # Si el webhook devolvió un error (ej. validación HTTP fallida)
            if response.status_code != 200:
                error_detail = response.text
                return f"Error en la llamada del webhook (Código HTTP {response.status_code}): {error_detail}"

            # 5. Definimos un generador interno que parsea la respuesta SSE en tiempo real
            def sse_generator() -> Generator[str, None, None]:
                # iter_lines procesa el streaming línea por línea
                for line in response.iter_lines():
                    if line:
                        linea_decodificada = line.decode('utf-8')
                        
                        # El formato estándar de SSE es 'data: <token_o_mensaje>'
                        if linea_decodificada.startswith("data: "):
                            # Extraemos el fragmento de texto (quitando los primeros 6 caracteres 'data: ')
                            token_content = linea_decodificada[6:]
                            yield token_content

            # Devolvemos el generador para que Open WebUI imprima los tokens en tiempo real
            return sse_generator()

        except requests.exceptions.Timeout:
            return "Error: Se agotó el tiempo de espera de la solicitud (Timeout en comunicación Pod a Pod)."
        except requests.exceptions.RequestException as e:
            return f"Error al comunicarse con el webhook del agente (Pod a Pod en OpenShift): {str(e)}"
