# Backend image (baseline model). Artefacts are mounted, not baked in, because the corpus is
# CC BY-SA and regenerable:  docker run -p 8000:8000 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" -v "$PWD/reports:/app/reports" moviesense-api
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY configs ./configs
ENV PYTHONPATH=/app/src MOVIESENSE_ROOT=/app MOVIESENSE_CORS_ORIGINS=http://localhost:5173
EXPOSE 8000
CMD ["uvicorn", "moviesense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
