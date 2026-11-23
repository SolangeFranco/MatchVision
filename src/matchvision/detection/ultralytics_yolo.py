"""Ultralytics YOLO detector adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from matchvision.detection.base import Detection


class UltralyticsYOLODetector:
    """Run a pretrained or custom Ultralytics YOLO detector on BGR frames."""

    def __init__(
        self,
        model: str | Path = "yolo11n.pt",
        *,
        confidence: float = 0.25,
        image_size: int = 1280,
        device: str | None = None,
        classes: list[int] | None = None,
        model_source: str | None = None,
        model_license: str | None = None,
        pretrained_on: str | None = None,
    ) -> None:
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if image_size < 32:
            raise ValueError("image_size must be at least 32")

        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Vision dependencies are missing. Install them with "
                "`python -m pip install -e \".[vision]\"`."
            ) from error

        self._ultralytics_version = ultralytics.__version__
        self._model_argument = str(model)
        if self._model_argument == "yolo11n.pt":
            model_source = model_source or (
                "https://github.com/ultralytics/ultralytics (official YOLO11n COCO checkpoint)"
            )
            model_license = model_license or (
                "AGPL-3.0; check Ultralytics commercial license for other use"
            )
            pretrained_on = pretrained_on or "COCO"
        elif not all((model_source, model_license, pretrained_on)):
            raise ValueError(
                "Custom model files require model_source, model_license, and "
                "pretrained_on provenance."
            )
        self._model_source = model_source
        self._model_license = model_license
        self._pretrained_on = pretrained_on
        self._model = YOLO(self._model_argument)
        self._confidence = confidence
        self._image_size = image_size
        self._device = device
        self._classes = classes
        self._class_names = {
            int(class_id): str(name)
            for class_id, name in self._model.names.items()
        }
        if self._classes is not None:
            unknown_class_ids = set(self._classes) - set(self._class_names)
            if unknown_class_ids:
                raise ValueError(
                    f"Model has no class IDs {sorted(unknown_class_ids)}"
                )
        if self._classes is None:
            specialized_labels = {
                name.lower().replace("_", " ")
                for name in self._class_names.values()
            }
            known_labels = {
                "person",
                "player",
                "football player",
                "referee",
                "goalkeeper",
                "ball",
                "sports ball",
                "football",
                "soccer ball",
            }
            if specialized_labels & known_labels:
                self._classes = [
                    class_id
                    for class_id, name in self._class_names.items()
                    if name.lower().replace("_", " ") in known_labels
                ]

    @property
    def model_name(self) -> str:
        return Path(self._model_argument).name

    @property
    def model_version(self) -> str:
        return self._ultralytics_version

    @property
    def model_source(self) -> str:
        return str(self._model_source)

    @property
    def model_license(self) -> str:
        return str(self._model_license)

    @property
    def pretrained_on(self) -> str:
        return str(self._pretrained_on)

    @property
    def class_names(self) -> dict[int, str]:
        if self._classes is None:
            return self._class_names.copy()
        return {
            class_id: name
            for class_id, name in self._class_names.items()
            if class_id in self._classes
        }

    @property
    def checkpoint_path(self) -> Path | None:
        weights = getattr(self._model, "ckpt_path", None)
        if not weights:
            return None
        path = Path(weights).expanduser()
        return path.resolve() if path.is_file() else None

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "confidence": self._confidence,
            "image_size": self._image_size,
            "device": self._device or "auto",
            "class_ids": self._classes,
        }

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (height, width, 3)")

        result = self._model.predict(
            source=frame,
            conf=self._confidence,
            imgsz=self._image_size,
            device=self._device,
            classes=self._classes,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(np.int32)
        return [
            Detection(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=float(confidence),
                class_id=int(class_id),
                label=self._class_names[int(class_id)],
            )
            for box, confidence, class_id in zip(
                xyxy, confidences, class_ids, strict=True
            )
        ]
