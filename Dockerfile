FROM python:3.11-slim

# Instalar dependencias de sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Exponer puerto de FastAPI
EXPOSE 8000

# Ejecutar la API
CMD ["python", "main.py"]