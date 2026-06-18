FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY services/adios-inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/adios-inference/ ./services/adios-inference/
EXPOSE 8010
CMD ["uvicorn", "services.adios-inference.main:app", "--host", "0.0.0.0", "--port", "8010"]
