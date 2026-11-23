"""Calibración manual de una vista de cancha con homografía planar."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


class CalibrationError(ValueError):
    """Error de correspondencias o geometría en una calibración."""


Point2D = tuple[float, float]
Matrix3x3 = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class PitchCalibration:
    """Homografía de una vista de cámara al plano de cancha especificado."""

    image_size: tuple[int, int]
    image_points: tuple[Point2D, ...]
    pitch_points: tuple[Point2D, ...]
    homography: Matrix3x3
    inlier_mask: tuple[bool, ...]
    reprojection_rmse: float
    units: str
    scale_verified: bool
    ransac_reprojection_threshold: float
    reference_frame_index: int | None = None

    @property
    def inlier_count(self) -> int:
        return sum(self.inlier_mask)

    def project_points(
        self, image_points: Sequence[Point2D] | np.ndarray
    ) -> np.ndarray:
        """Proyecta puntos de imagen a las unidades de cancha declaradas."""
        points = _points_array(image_points, "image_points", minimum_count=1)
        width, height = self.image_size
        if (
            np.any(points[:, 0] < 0)
            or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] < 0)
            or np.any(points[:, 1] >= height)
        ):
            raise CalibrationError("Hay puntos fuera de los límites de la imagen")
        mapped = cv2.perspectiveTransform(
            points.astype(np.float64).reshape(-1, 1, 2),
            np.asarray(self.homography, dtype=np.float64),
        ).reshape(-1, 2)
        if not np.isfinite(mapped).all():
            raise CalibrationError(
                "La homografía proyecta un punto al infinito o a un valor inválido"
            )
        return mapped

    def project_player_footpoint(
        self, box_xyxy: Sequence[float]
    ) -> tuple[float, float]:
        """Proyecta el centro inferior de una caja de jugador."""
        if len(box_xyxy) != 4:
            raise CalibrationError("La caja debe tener el formato x1, y1, x2, y2")
        box = np.asarray(box_xyxy, dtype=np.float64)
        if not np.isfinite(box).all():
            raise CalibrationError("La caja contiene coordenadas no finitas")
        x1, y1, x2, y2 = (float(value) for value in box)
        width, height = self.image_size
        if x2 <= x1 or y2 <= y1:
            raise CalibrationError("La caja de jugador debe tener área positiva")
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise CalibrationError("La caja de jugador queda fuera de la imagen")
        foot_y = min(float(y2), float(np.nextafter(height, 0)))
        footpoint = self.project_points([[(x1 + x2) / 2, foot_y]])[0]
        return float(footpoint[0]), float(footpoint[1])

    def to_dict(self) -> dict[str, object]:
        """Serializa la calibración y sus unidades sin afirmar su exactitud."""
        return {
            "schema_version": 1,
            "image_size": list(self.image_size),
            "image_points": [list(point) for point in self.image_points],
            "pitch_points": [list(point) for point in self.pitch_points],
            "homography_image_to_pitch": [
                list(row) for row in self.homography
            ],
            "inlier_mask": list(self.inlier_mask),
            "inlier_count": self.inlier_count,
            "reprojection_rmse": self.reprojection_rmse,
            "units": self.units,
            "scale_verified": self.scale_verified,
            "ransac_reprojection_threshold": (
                self.ransac_reprojection_threshold
            ),
            "reference_frame_index": self.reference_frame_index,
            "limitations": [
                "Homografía válida solo para el plano y la vista de cámara calibrados.",
                "Las posiciones de jugadores deben aproximarse con puntos de contacto "
                "con el suelo, no con el centro de sus cajas.",
                (
                    "Las unidades de salida son coordenadas; su escala física no "
                    "está verificada."
                    if not self.scale_verified
                    else "La escala se declaró verificada por quien aportó las medidas."
                ),
                "No se estima movimiento de cámara ni se interpolan cambios de plano.",
            ],
        }


def fit_pitch_calibration(
    image_points: Sequence[Point2D] | np.ndarray,
    pitch_points: Sequence[Point2D] | np.ndarray,
    *,
    image_size: tuple[int, int],
    units: str,
    scale_verified: bool,
    ransac_reprojection_threshold: float,
    reference_frame_index: int | None = None,
    maximum_reprojection_rmse: float | None = None,
) -> PitchCalibration:
    """Ajusta una homografía RANSAC desde correspondencias manuales 2D.

    El umbral RANSAC y RMSE usan las unidades declaradas para los puntos de
    cancha. La verificación de escala la debe confirmar el proveedor de los
    puntos; nunca se infiere de la geometría observada.
    """
    image_size = _validate_image_size(image_size)
    if not isinstance(units, str) or not units.strip():
        raise CalibrationError("units debe declarar las unidades de cancha")
    if not isinstance(scale_verified, bool):
        raise CalibrationError("scale_verified debe ser booleano")
    if (
        not math.isfinite(ransac_reprojection_threshold)
        or ransac_reprojection_threshold <= 0
    ):
        raise CalibrationError(
            "ransac_reprojection_threshold debe ser finito y positivo"
        )
    if maximum_reprojection_rmse is not None and (
        not math.isfinite(maximum_reprojection_rmse)
        or maximum_reprojection_rmse <= 0
    ):
        raise CalibrationError(
            "maximum_reprojection_rmse debe ser finito y positivo"
        )
    if reference_frame_index is not None and (
        not isinstance(reference_frame_index, int)
        or isinstance(reference_frame_index, bool)
        or reference_frame_index < 0
    ):
        raise CalibrationError("reference_frame_index debe ser un entero no negativo")

    image_array = _points_array(image_points, "image_points", minimum_count=1)
    pitch_array = _points_array(pitch_points, "pitch_points", minimum_count=1)
    if len(image_array) != len(pitch_array):
        raise CalibrationError(
            "Cada punto de imagen debe corresponder a un punto de cancha"
        )
    if len(image_array) < 4:
        raise CalibrationError(
            "Se requieren al menos cuatro pares de correspondencias 2D"
        )
    width, height = image_size
    if (
        np.any(image_array[:, 0] < 0)
        or np.any(image_array[:, 0] >= width)
        or np.any(image_array[:, 1] < 0)
        or np.any(image_array[:, 1] >= height)
    ):
        raise CalibrationError(
            "Los puntos de imagen deben quedar dentro de la imagen de referencia"
        )

    matrix, mask = cv2.findHomography(
        image_array,
        pitch_array,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reprojection_threshold,
    )
    if matrix is None or mask is None:
        raise CalibrationError(
            "No se pudo ajustar la homografía; revisa las correspondencias"
        )
    inliers = mask.reshape(-1).astype(bool)
    if int(np.count_nonzero(inliers)) < 4:
        raise CalibrationError(
            "La homografía requiere al menos cuatro correspondencias inlier"
        )
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise CalibrationError("La homografía calculada es singular o no finita")

    projected = cv2.perspectiveTransform(
        image_array[inliers].astype(np.float64).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    if not np.isfinite(projected).all():
        raise CalibrationError("La homografía produce reproyecciones no finitas")
    errors = np.linalg.norm(projected - pitch_array[inliers], axis=1)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    if maximum_reprojection_rmse is not None and rmse > maximum_reprojection_rmse:
        raise CalibrationError(
            f"El RMSE de reproyección ({rmse:.6g} {units}) supera el máximo "
            f"declarado ({maximum_reprojection_rmse:.6g} {units})"
        )

    matrix /= matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else np.linalg.norm(matrix)
    return PitchCalibration(
        image_size=image_size,
        image_points=tuple(map(tuple, image_array.tolist())),
        pitch_points=tuple(map(tuple, pitch_array.tolist())),
        homography=tuple(map(tuple, matrix.tolist())),
        inlier_mask=tuple(bool(value) for value in inliers),
        reprojection_rmse=rmse,
        units=units.strip(),
        scale_verified=scale_verified,
        ransac_reprojection_threshold=float(ransac_reprojection_threshold),
        reference_frame_index=reference_frame_index,
    )


def _points_array(
    points: Sequence[Point2D] | np.ndarray,
    name: str,
    *,
    minimum_count: int,
) -> np.ndarray:
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{name} debe ser una lista numérica Nx2") from error
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < minimum_count:
        raise CalibrationError(
            f"{name} debe contener al menos {minimum_count} puntos 2D"
        )
    if not np.isfinite(array).all():
        raise CalibrationError(f"{name} contiene coordenadas no finitas")
    return np.ascontiguousarray(array)


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(image_size, (tuple, list))
        or len(image_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in image_size
        )
    ):
        raise CalibrationError("image_size debe ser (ancho, alto) en píxeles")
    return int(image_size[0]), int(image_size[1])
