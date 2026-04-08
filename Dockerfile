# =============================================================================
# Water Surface Trash Collector — Hugging Face Spaces (Gradio GUI)
# =============================================================================
FROM python:3.10-slim

WORKDIR /app/env

# Install system deps + fonts for PIL text rendering
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

# Expose port (HF Spaces default)
EXPOSE 7860

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run the Gradio web GUI
CMD ["python", "web_gui.py"]
