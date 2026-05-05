# Usamos una imagen ligera de Python
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Instalamos dependencias del sistema necesarias para networking (opcional pero recomendado)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copiamos el archivo de requerimientos primero para aprovechar el cache de capas
COPY requirements.txt .

# Instalamos las librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código (mcp_client.py, agent.py, main.py)
COPY . .

# Exponemos el puerto que definimos en el ConfigMap (estándar 8080 para OpenShift)
EXPOSE 8080

# Comando para arrancar la aplicación
# Usamos uvicorn para manejar las peticiones asíncronas del agente
CMD ["python", "main.py"]