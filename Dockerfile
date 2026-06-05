FROM python:3.11-slim

# Variables de entorno para optimizar Python en Docker
# - PYTHONDONTWRITEBYTECODE: Evita que python escriba archivos .pyc en disco
# - PYTHONUNBUFFERED: Fuerza a que los logs se envíen directamente al stdout sin búfer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Copiamos requirements.txt e instalamos dependencias de manera directa con acceso a internet
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiamos el código fuente de la aplicación
COPY app/ ./app

# Exponemos el puerto de escucha (estándar en OpenShift es 8080)
EXPOSE 8080

# En OpenShift (incluido OpenShift 3.4), los contenedores se ejecutan por defecto con un UID aleatorio
# asignado dinámicamente que pertenece al grupo raíz (GID 0).
# Para evitar errores de permisos, es una regla estricta dar propiedad y permisos de lectura/escritura
# al grupo raíz (0) sobre el directorio de la aplicación.
RUN chown -R 1001:0 /app && \
    chmod -R g+rwX /app

# Cambiamos al usuario no privilegiado (ID 1001) para entornos Kubernetes estándar
USER 1001

# Ejecutamos el servidor FastAPI con Uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
