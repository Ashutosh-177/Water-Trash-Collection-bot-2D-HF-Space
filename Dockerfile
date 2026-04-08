# =============================================================================
# Water Surface Trash Collector — Docker Image
# =============================================================================
FROM python:3.10-slim

WORKDIR /app/env

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.0.0" \
    "gradio>=4.0.0" \
    "Pillow>=10.0.0"

# Expose port (OpenEnv standard is 8000)
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=false
# We set ENABLE_WEB_INTERFACE to false so OpenEnv does not mount its default web interface.
# We will mount our own custom Gradio interface in server/app.py instead.

ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server natively which exposes all OpenEnv endpoints!
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
