FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src ./src
COPY data/benchmark_corpus ./data/benchmark_corpus

RUN pip install --no-cache-dir -e ".[api,eval]"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "cag.api.upload:app", "--host", "0.0.0.0", "--port", "8000"]

