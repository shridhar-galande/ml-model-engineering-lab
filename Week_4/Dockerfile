FROM python:3.11-slim

WORKDIR /app

ARG MODEL_PATH
COPY ${MODEL_PATH}/ /model/

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /model/requirements.txt

EXPOSE 5000

CMD ["mlflow", "models", "serve", "--model-uri", "/model", "--host", "0.0.0.0", "--port", "5000", "--env-manager", "local"]
