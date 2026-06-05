"""
Herramienta Personalizada para Open WebUI - Conexión con el Agente SRE

Este script está diseñado para ser copiado y configurado directamente en el panel de control 
de Open WebUI (Sección de Herramientas/Tools). Permite a la interfaz de usuario invocar 
de forma dinámica al Agente SRE basado en LangGraph que se ejecuta en el mismo clúster.

REGLA DE RED Y COMUNICACIÓN INTERNA (POD A POD):
------------------------------------------------
Este script asume que la comunicación es interna de Pod a Pod dentro del mismo proyecto
(namespace) de OpenShift. Utiliza el servicio DNS de Kubernetes interno para resolver 
la IP del webhook del agente de forma segura, evitando exponer el endpoint del agente
al exterior del clúster de forma innecesaria.
"""

import requests
import json
from typing import Generator, Any

class Tools:
    def __init__(self):
        # Definición de variables de configuración (Valves en OpenUI)
        # Esto permite cambiar la URL de conexión desde la interfaz web de Open WebUI.
        self.valves = {
            # URL interna del servicio del webhook en OpenShift
            "AGENT_WEBHOOK_URL": "http://agente-webhook:8000/webhook",
            
            # URL utilizada para pasar la validación inicial asíncrona del webhook (HTTP 200 OK)
            "DEFAULT_VALIDATION_URL": "http://example.com"
        }

    def consultar_agente_sre(self, mensaje_consulta: str) -> str:
        """
        Envía una consulta técnica de SRE o Kubernetes al Agente Inteligente.
        
        :param mensaje_consulta: Pregunta o instrucción técnica (ej. 'Lista el estado de los pods en el namespace actual').
        :return: Respuesta acumulada del agente SRE.
        """
        # Leemos los parámetros configurados en las válvulas
        url = self.valves.get("AGENT_WEBHOOK_URL", "http://agente-webhook:8000/webhook")
        validation_url = self.valves.get("DEFAULT_VALIDATION_URL", "http://example.com")
        
        # Estructuramos la carga útil JSON que espera nuestro webhook en FastAPI
        # Enviamos un session_id estático o generado para mantener el AgentState (historial humano/IA)
        payload = {
            "url": validation_url,
            "message": mensaje_consulta,
            "session_id": "openwebui-user-conversation"
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Informamos al usuario en Open WebUI de que se está conectando con el backend
        print(f"Enviando consulta interna a {url}...")
        
        try:
            # Hacemos la petición POST por streaming.
            # stream=True nos permite procesar la respuesta línea por línea
            # conforme el modelo de lenguaje va generando los tokens (SSE).
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=45 # Timeout largo para esperar la respuesta del LLM
            )
            
            # Si el webhook devolvió un error (ej. URL de validación no responde -> 400 Bad Request)
            if response.status_code != 200:
                error_msg = response.text
                return f"Error en la llamada del webhook (Código HTTP {response.status_code}): {error_msg}"
            
            # Procesamos el streaming de Server-Sent Events (SSE) y acumulamos la respuesta
            respuesta_acumulada = ""
            
            # iter_lines procesa el streaming línea por línea
            for line in response.iter_lines():
                if line:
                    linea_decodificada = line.decode('utf-8')
                    
                    # El formato estándar de SSE es 'data: <token_o_mensaje>'
                    if linea_decodificada.startswith("data: "):
                        # Extraemos el fragmento de texto (quitando los primeros 6 caracteres 'data: ')
                        token_content = linea_decodificada[6:]
                        
                        # Acumulamos el token en la respuesta final
                        respuesta_acumulada += token_content
            
            if not respuesta_acumulada:
                return "El agente devolvió una respuesta vacía."
                
            return respuesta_acumulada

        except requests.exceptions.Timeout:
            return "Error: Se agotó el tiempo de espera de la solicitud (Timeout de Pod a Pod)."
        except requests.exceptions.RequestException as e:
            return f"Error al intentar comunicarse con el webhook del Agente (Pod a Pod en OpenShift): {str(e)}"
