"""Local OSNet-AIN person-ReID embeddings with verified model weights."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_NAME = "osnet_ain_x1_0"
MODEL_SHA256 = "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f"
MODEL_FILENAME = (
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_"
    "fb10_softmax_labsmth_flip_jitter.pth"
)
MODEL_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
    f"{MODEL_FILENAME}?download=true"
)
MODEL_SIZE_BYTES = 17_293_009


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine similarity of finite, non-zero embeddings."""
    first_vector = np.asarray(first, dtype=np.float32).reshape(-1)
    second_vector = np.asarray(second, dtype=np.float32).reshape(-1)
    if first_vector.shape != second_vector.shape or first_vector.size == 0:
        raise ValueError("Los embeddings deben tener la misma dimensión no vacía.")
    if not np.isfinite(first_vector).all() or not np.isfinite(second_vector).all():
        raise ValueError("Los embeddings deben contener solo valores finitos.")
    first_norm = float(np.linalg.norm(first_vector))
    second_norm = float(np.linalg.norm(second_vector))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        raise ValueError("No se puede calcular similitud de un embedding nulo.")
    dot_product = float(np.dot(first_vector, second_vector) / (first_norm * second_norm))
    return float(np.clip(dot_product, -1.0, 1.0))


def resolve_device(requested: str = "auto") -> str:
    """Resolve auto to CUDA, MPS, or CPU without silently masking errors."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "La extracción OSNet requiere el extra opcional «reid»."
        ) from error
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def extract_person_crop(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    minimum_width: int = 8,
    minimum_height: int = 16,
) -> np.ndarray | None:
    """Clip a player box to the frame and return its BGR crop."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Se esperaba un fotograma BGR con forma (alto, ancho, 3).")
    if len(box) != 4 or not np.isfinite(box).all():
        raise ValueError("La caja debe tener cuatro coordenadas finitas.")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("La caja debe tener área positiva.")
    height, width = frame.shape[:2]
    left = max(0, min(width, int(np.floor(x1))))
    top = max(0, min(height, int(np.floor(y1))))
    right = max(0, min(width, int(np.ceil(x2))))
    bottom = max(0, min(height, int(np.ceil(y2))))
    if right - left < minimum_width or bottom - top < minimum_height:
        return None
    return frame[top:bottom, left:right].copy()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OSNetEmbeddingExtractor:
    """Batch OSNet-AIN inference; outputs L2-normalized appearance descriptors."""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("El tamaño del lote debe ser positivo.")
        self.weights_path = Path(weights_path).expanduser().resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"No se encontraron los pesos OSNet: {self.weights_path}. "
                "Descárguelos con matchvision-download-reid-model."
            )
        actual_sha256 = _sha256_file(self.weights_path)
        if actual_sha256 != MODEL_SHA256:
            raise ValueError(
                "El SHA-256 de los pesos OSNet no coincide con el modelo esperado."
            )

        try:
            import torch
            from torchreid.reid.models import build_model
        except ImportError as error:
            raise RuntimeError(
                "Instale las dependencias con «pip install -e .[reid]»."
            ) from error

        self.device = resolve_device(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(
                f"El dispositivo solicitado no está disponible: {self.device}"
            )
        if self.device.startswith("mps") and not torch.backends.mps.is_available():
            raise ValueError(
                f"El dispositivo solicitado no está disponible: {self.device}"
            )
        self.batch_size = batch_size
        self._torch = torch
        self._torch_version = torch.__version__
        self._model = build_model(
            MODEL_NAME,
            num_classes=1,
            pretrained=False,
            use_gpu=self.device.startswith("cuda"),
        )
        checkpoint = torch.load(
            self.weights_path, map_location="cpu", weights_only=True
        )
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, dict):
            raise ValueError("El checkpoint OSNet no contiene un state_dict válido.")
        model_state = self._model.state_dict()
        compatible: dict[str, Any] = {}
        matched_numel = 0
        expected_numel = 0
        for key, target in model_state.items():
            if not key.startswith("classifier."):
                expected_numel += target.numel()
        for key, value in state.items():
            normalized_key = key.removeprefix("module.")
            target = model_state.get(normalized_key)
            if target is not None and target.shape == value.shape:
                compatible[normalized_key] = value
                if not normalized_key.startswith("classifier."):
                    matched_numel += target.numel()
        coverage = matched_numel / max(expected_numel, 1)
        if coverage < 0.95:
            raise ValueError(
                f"Los pesos OSNet solo cubren {coverage:.1%} de la arquitectura."
            )
        self._model.load_state_dict(compatible, strict=False)
        self._model.to(self.device)
        self._model.eval()
        self._mean = torch.tensor(
            (0.485, 0.456, 0.406), dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            (0.229, 0.224, 0.225), dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.inference_seconds = 0.0
        self.images_processed = 0

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "model": MODEL_NAME,
            "weights": self.weights_path.name,
            "weights_sha256": MODEL_SHA256,
            "pretrained_on": "MSMT17",
            "input_size": [256, 128],
            "embedding_dimensions": 512,
            "device": self.device,
            "torch_version": self._torch_version,
            "batch_size": self.batch_size,
            "similarity": "coseno; descriptor de apariencia, no probabilidad",
        }

    def encode(self, crops_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Encode BGR crops in bounded batches and return unit-norm vectors."""
        if not crops_bgr:
            return []
        images: list[np.ndarray] = []
        for crop in crops_bgr:
            if crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
                raise ValueError("Cada recorte debe ser una imagen BGR no vacía.")
            resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
            images.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        results: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch_images = images[start : start + self.batch_size]
            array = np.stack(batch_images).astype(np.float32) / 255.0
            tensor = self._torch.from_numpy(array).permute(0, 3, 1, 2).to(self.device)
            tensor = (tensor - self._mean) / self._std
            started = time.perf_counter()
            with self._torch.inference_mode():
                features = self._model(tensor)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                features = self._torch.nn.functional.normalize(features.float(), dim=1)
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize(self.device)
            self.inference_seconds += time.perf_counter() - started
            vectors = features.detach().cpu().numpy().astype(np.float32)
            if vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise RuntimeError("OSNet devolvió embeddings no válidos.")
            results.extend(vectors)
        self.images_processed += len(results)
        return results
