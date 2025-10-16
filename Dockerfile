FROM python:3.10-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py train.py ./
RUN python train.py

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app/model_weights.pth    .
COPY --from=builder /app/scaler_mean.npy      .
COPY --from=builder /app/scaler_scale.npy     .
COPY model.py server.py requirements.txt      .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
