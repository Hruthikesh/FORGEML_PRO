FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY forgeml ./forgeml
RUN pip install --no-cache-dir ".[boosting]"
COPY frontend ./frontend
COPY scripts ./scripts
COPY artifacts ./artifacts
RUN useradd -m forge && chown -R forge:forge /app
USER forge
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"
CMD ["uvicorn", "forgeml.api:app", "--host", "0.0.0.0", "--port", "8000"]
