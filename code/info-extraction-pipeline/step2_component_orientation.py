from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ORIENTATION_CLASS_TO_IDX = {"u": 0, "r": 1, "d": 2, "l": 3}
ORIENTATION_IDX_TO_CLASS = {idx: label for label, idx in ORIENTATION_CLASS_TO_IDX.items()}
ORIENTATION_REQUIRED_CLASSES = {
    "nmos",
    "nmos-bulk",
    "pmos",
    "pmos-bulk",
    "npn",
    "pnp",
    "diode",
    "amplifier",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_orientation_model(weights_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(weights_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", ORIENTATION_CLASS_TO_IDX)
    if class_to_idx != ORIENTATION_CLASS_TO_IDX:
        raise ValueError(f"Unexpected orientation class mapping in checkpoint: {class_to_idx}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(ORIENTATION_CLASS_TO_IDX))
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def orientation_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )




def classify_orientation(
    crop: Image.Image,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
        tensor = transform(crop.convert("RGB")).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
        orientation_id = int(torch.argmax(probs).item())
        orientation = ORIENTATION_IDX_TO_CLASS[orientation_id]
        return {
            "orientation": orientation,
            "orientation_id": orientation_id,
            "orientation_confidence": float(probs[orientation_id].item()),
            "orientation_probabilities": {
                ORIENTATION_IDX_TO_CLASS[idx]: float(prob.item()) for idx, prob in enumerate(probs)
            },
        }


def safe_component_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


