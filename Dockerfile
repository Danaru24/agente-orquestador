FROM python:3.11-slim

WORKDIR /app

# Instalación de dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instalación de dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SOLUCIÓN AL ERROR: Copiar el contenido de la subcarpeta 'app' al directorio actual
COPY app/ .

EXPOSE 8080

# Ahora 'main.py' estará en la ruta raíz del WORKDIR
CMD ["python", "main.py"]
