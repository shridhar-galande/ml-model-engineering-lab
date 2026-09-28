FROM python:3.12-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY app.py .
ARG MODEL_VERSION=v1
COPY models/model_${MODEL_VERSION}.joblib model.joblib
ENV MODEL_VERSION=${MODEL_VERSION} MODEL_PATH=model.joblib
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
