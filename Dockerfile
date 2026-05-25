FROM python:3.13-slim AS runtime

ENV HOST=0.0.0.0 \
    PORT=8080 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md config.example.yaml ./
COPY claude_code_proxy ./claude_code_proxy
COPY main.py ./

RUN pip install --no-cache-dir .

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).read()"

CMD ["uvicorn", "claude_code_proxy.server:app", "--host", "0.0.0.0", "--port", "8080"]
