FROM python:3.11-slim

WORKDIR /app

ARG MODEL_PATH=_docker_model
COPY ${MODEL_PATH}/ /model/
COPY src/ /app/src/

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /model/requirements.txt \
    && python -m pip install --no-cache-dir "fastapi>=0.110,<1" "uvicorn[standard]>=0.27,<1" "prometheus-client>=0.20,<1"

ENV PYTHONPATH=/app
ENV MODEL_PATH=/model

EXPOSE 8000

CMD ["uvicorn", "src.monitoring_service:app", "--host", "0.0.0.0", "--port", "8000"]
