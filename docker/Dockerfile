FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install uv && uv pip install --system -e ".[api,cloud]"

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ src/
COPY .env .env
CMD ["uvicorn", "rag_forge.api.app:app", "--host", "0.0.0.0", "--port", "8000"]