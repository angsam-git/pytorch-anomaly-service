import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from model import Autoencoder

# ————— Logging —————
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anomaly-service")

# ————— Load artifacts —————
try:
    model_weights = torch.load("model_weights.pth", map_location="cpu")  # removed weights_only
    scaler_params = np.load("scaler_params.npz")
    scaler_mean, scaler_scale = scaler_params["mean"], scaler_params["scale"]
    input_dim = scaler_mean.shape[0]
    logger.info(f"Loaded model and scaler (input_dim={input_dim})")
except Exception as e:
    logger.exception("Failed to load model or scaler")
    raise

# ————— Model init —————
model = Autoencoder(input_dim=input_dim, hidden_dims=[128, 64], latent_dim=16, dropout=0.2)
model.load_state_dict(model_weights)
model.eval()

# ————— Threshold —————
# Ideally computed statistically after training, e.g. mean + 3σ
try:
    THRESHOLD = float(np.loadtxt("threshold.txt"))  # optional external config
except Exception:
    THRESHOLD = 0.5
logger.info(f"Using anomaly threshold: {THRESHOLD}")

# ————— FastAPI —————
app = FastAPI()

class LogEntryRequest(BaseModel):
    timestamp: int
    endpoint: str
    method: str
    status: int
    responseTimeMs: int
    queryCount: int
    headerCount: int
    responseSizeBytes: int
    requestId: str

class PredictResponse(BaseModel):
    reconstruction_error: float
    is_anomaly: bool


def scale_input(x: list[float], mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - mean) / scale


@app.post("/predict", response_model=PredictResponse)
def predict(req: LogEntryRequest):
    try:
        start = time.time()

        features = [
            float(req.responseTimeMs),
            float(req.status),
            float(req.queryCount),
            float(req.headerCount),
            float(req.responseSizeBytes),
        ]

        if len(features) != input_dim:
            raise ValueError(f"Expected {input_dim} features, got {len(features)}")

        x_scaled = scale_input(features, scaler_mean, scaler_scale)
        xt = torch.from_numpy(x_scaled).unsqueeze(0)

        logger.info(f"[{req.requestId}] Feature vector: {features}")

        # Inference
        with torch.inference_mode():
            recon = model(xt)
            err = torch.mean((xt - recon) ** 2).item()

        is_anom = err > THRESHOLD
        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"[{req.requestId}] error={err:.6f}, anomaly={is_anom}, "
            f"inference_time={elapsed_ms:.2f} ms"
        )

        return PredictResponse(reconstruction_error=err, is_anomaly=is_anom)

    except Exception as e:
        logger.exception(f"[{req.requestId}] Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
