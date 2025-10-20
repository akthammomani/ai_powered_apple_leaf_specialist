#============================================================#
#                     src/inference.py                       #
#============================================================#
# Author      : Aktham Almomani                              #
# Created     : 2025-10-01                                   #
# Version     : V1.0.0                                       #
#============================================================#


import io, json, base64
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Matching class order used in training:
LABELS = ["black_rot", "healthy", "rust", "scab"]

IMG_SIZE = 256  # same as training

_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),          
])

def _build_model(num_classes=len(LABELS)):
    m = models.resnet18(weights=None)      
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.eval()
    return m

def model_fn(model_dir):
    model = _build_model()
    state = torch.load(Path(model_dir) / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        obj = json.loads(request_body)
        b = base64.b64decode(obj["b64"])
        img = Image.open(io.BytesIO(b)).convert("RGB")
        return _tf(img).unsqueeze(0)
    if request_content_type in ("image/jpeg", "image/png"):
        img = Image.open(io.BytesIO(request_body)).convert("RGB")
        return _tf(img).unsqueeze(0)
    raise ValueError(f"Unsupported content type: {request_content_type}")

@torch.inference_mode()
def predict_fn(input_data, model):
    logits = model(input_data)
    probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    idx = int(torch.argmax(logits, dim=1).item())
    return {"label": LABELS[idx], "probs": probs, "labels": LABELS}

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
