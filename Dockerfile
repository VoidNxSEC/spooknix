# Dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

# Dependências do sistema (ffmpeg para processamento de áudio)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python (torch já presente na base image)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY src/ ./src/
COPY web/ ./web/

# Criar diretórios de output
RUN mkdir -p /app/outputs/transcripts /app/outputs/subtitles

# Porta para servidor WebSocket
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" \
    || exit 1


# Default: Server mode
CMD ["python", "-m", "src.server"]
