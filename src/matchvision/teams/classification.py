"""Agrupamiento no supervisado y conservador de camisetas en tracks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

AssignmentState = Literal[
    "assigned",
    "ambiguous",
    "outlier",
    "insufficient_samples",
    "insufficient_data",
    "manual",
]


@dataclass(frozen=True, slots=True)
class TeamAssignment:
    """Equipo estimado para un track o una identidad proporcionados."""

    identity_id: str
    team_id: str | None
    state: AssignmentState
    confidence: float | None
    sample_count: int


class UniformTeamClassifier:
    """Agrupa descriptores del torso en dos equipos sin fusionar personas.

    ``identity_id`` solo agrupa observaciones; este clasificador nunca lo
    deduce ni lo modifica. Los equipos se estiman por apariencia de camiseta,
    no son identidades individuales ni roles verificados.
    """

    def __init__(
        self,
        *,
        team_names: tuple[str, str] = ("equipo_a", "equipo_b"),
        minimum_samples: int = 3,
        minimum_cluster_size: int = 2,
        sample_interval_frames: int = 5,
        maximum_samples_per_identity: int = 20,
        minimum_cluster_separation: float = 0.18,
        minimum_assignment_margin: float = 0.12,
        maximum_assignment_distance: float = 0.62,
    ) -> None:
        if len(team_names) != 2 or not all(
            isinstance(name, str) and name.strip() for name in team_names
        ):
            raise ValueError("team_names debe contener dos etiquetas no vacías")
        if team_names[0] == team_names[1]:
            raise ValueError("team_names debe contener dos etiquetas distintas")
        if minimum_samples < 1 or minimum_cluster_size < 1:
            raise ValueError("los mínimos de muestras y tamaño de grupo deben ser positivos")
        if sample_interval_frames < 1 or maximum_samples_per_identity < 1:
            raise ValueError("el intervalo y el límite de muestras deben ser positivos")
        for value, name in (
            (minimum_cluster_separation, "minimum_cluster_separation"),
            (minimum_assignment_margin, "minimum_assignment_margin"),
            (maximum_assignment_distance, "maximum_assignment_distance"),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} debe ser finito y estar entre 0 y 1")

        self._team_names = team_names
        self._minimum_samples = minimum_samples
        self._minimum_cluster_size = minimum_cluster_size
        self._sample_interval = sample_interval_frames
        self._maximum_samples = maximum_samples_per_identity
        self._minimum_separation = minimum_cluster_separation
        self._minimum_margin = minimum_assignment_margin
        self._maximum_distance = maximum_assignment_distance
        self._samples: dict[str, list[np.ndarray]] = defaultdict(list)
        self._seen_identities: set[str] = set()
        self._last_sample_frame: dict[str, int] = {}
        self._last_observed_frame: dict[str, int] = {}
        self._first_seen: dict[str, int] = {}
        self._manual_assignments: dict[str, str] = {}
        self._observations_seen = 0

    @property
    def team_names(self) -> tuple[str, str]:
        return self._team_names

    @property
    def minimum_samples(self) -> int:
        return self._minimum_samples

    @property
    def configuration(self) -> dict[str, int | float | list[str] | str]:
        return {
            "method": "histograma HSV del torso + histograma de brillo + k-means determinista de dos grupos",
            "feature": "histograma HSV de tono/saturación (18x4) e histograma de brillo (8 bins)",
            "team_names": list(self._team_names),
            "minimum_samples": self._minimum_samples,
            "minimum_cluster_size": self._minimum_cluster_size,
            "sample_interval_frames": self._sample_interval,
            "maximum_samples_per_identity": self._maximum_samples,
            "minimum_cluster_separation": self._minimum_separation,
            "minimum_assignment_margin": self._minimum_margin,
            "maximum_assignment_distance": self._maximum_distance,
            "manual_assignments": len(self._manual_assignments),
        }

    def observe(
        self,
        frame: np.ndarray,
        identity_id: str | int,
        box: tuple[float, float, float, float],
        frame_index: int,
    ) -> bool:
        """Guarda una muestra de apariencia espaciada; indica si se conservó."""
        key = str(identity_id).strip()
        if not key:
            raise ValueError("identity_id no puede estar vacío")
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 1:
            raise ValueError("frame_index debe ser un entero positivo y comenzar en 1")
        previous_observation = self._last_observed_frame.get(key)
        if previous_observation is not None and frame_index <= previous_observation:
            raise ValueError(
                f"Los fotogramas de la identidad {key} deben estar en orden creciente"
            )
        previous_frame = self._last_sample_frame.get(key)
        if (
            previous_frame is not None
            and frame_index - previous_frame < self._sample_interval
        ):
            _validate_uniform_input(frame, box)
            self._register_identity(key)
            self._last_observed_frame[key] = frame_index
            return False
        feature = _extract_uniform_feature(frame, box)
        self._register_identity(key)
        self._last_observed_frame[key] = frame_index
        if feature is None:
            return False
        samples = self._samples[key]
        if len(samples) >= self._maximum_samples:
            samples.pop(0)
        samples.append(feature)
        self._last_sample_frame[key] = frame_index
        return True

    def set_manual_assignment(self, identity_id: str | int, team_id: str) -> None:
        """Corrige manualmente la etiqueta de una identidad actual o futura."""
        key = str(identity_id).strip()
        if not key:
            raise ValueError("identity_id no puede estar vacío")
        if team_id not in self._team_names:
            raise ValueError(f"team_id debe ser uno de {self._team_names}")
        self._manual_assignments[key] = team_id

    def clear_manual_assignment(self, identity_id: str | int) -> None:
        """Elimina una corrección manual si existe."""
        self._manual_assignments.pop(str(identity_id).strip(), None)

    def classify(self) -> dict[str, TeamAssignment]:
        """Devuelve estimaciones prudentes para los ID observados o corregidos."""
        descriptors = {
            identity: _mean_histogram(samples)
            for identity, samples in self._samples.items()
            if len(samples) >= self._minimum_samples
        }
        all_ids = self._seen_identities | set(self._manual_assignments)
        result = {
            identity: TeamAssignment(
                identity_id=identity,
                team_id=None,
                state=(
                    "insufficient_samples"
                    if len(self._samples.get(identity, ())) < self._minimum_samples
                    else "insufficient_data"
                ),
                confidence=None,
                sample_count=len(self._samples.get(identity, ())),
            )
            for identity in all_ids
        }

        if len(descriptors) >= 2 * self._minimum_cluster_size:
            identities = sorted(
                descriptors,
                key=lambda identity: (self._first_seen.get(identity, 0), identity),
            )
            features = np.stack([descriptors[identity] for identity in identities])
            labels, centers = _two_cluster_kmeans(features)
            separation = _histogram_distance(centers[0], centers[1])
            if (
                np.count_nonzero(labels == 0) >= self._minimum_cluster_size
                and np.count_nonzero(labels == 1) >= self._minimum_cluster_size
                and separation >= self._minimum_separation
            ):
                first_identity_index = min(
                    range(len(identities)),
                    key=lambda index: (
                        self._first_seen.get(identities[index], 0),
                        identities[index],
                    ),
                )
                first_cluster = int(labels[first_identity_index])
                team_for_cluster = {
                    first_cluster: self._team_names[0],
                    1 - first_cluster: self._team_names[1],
                }
                for index, identity in enumerate(identities):
                    distances = np.asarray(
                        [
                            _histogram_distance(features[index], centers[cluster])
                            for cluster in (0, 1)
                        ]
                    )
                    nearest = int(np.argmin(distances))
                    nearest_distance = float(distances[nearest])
                    other_distance = float(distances[1 - nearest])
                    margin = (other_distance - nearest_distance) / max(
                        other_distance + nearest_distance, 1e-12
                    )
                    if nearest_distance > self._maximum_distance:
                        result[identity] = TeamAssignment(
                            identity, None, "outlier", None, len(self._samples[identity])
                        )
                    elif margin < self._minimum_margin:
                        result[identity] = TeamAssignment(
                            identity, None, "ambiguous", margin, len(self._samples[identity])
                        )
                    else:
                        result[identity] = TeamAssignment(
                            identity,
                            team_for_cluster[nearest],
                            "assigned",
                            margin,
                            len(self._samples[identity]),
                        )

        for identity, team_id in self._manual_assignments.items():
            result[identity] = TeamAssignment(
                identity,
                team_id,
                "manual",
                None,
                len(self._samples.get(identity, ())),
            )
        return dict(sorted(result.items()))

    def _register_identity(self, identity_id: str) -> None:
        self._observations_seen += 1
        self._seen_identities.add(identity_id)
        self._first_seen.setdefault(identity_id, self._observations_seen)


def _extract_uniform_feature(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray | None:
    coordinates = _validate_uniform_input(frame, box)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = coordinates
    left = max(0, min(width, int(np.floor(x1 + (x2 - x1) * 0.20))))
    right = max(0, min(width, int(np.ceil(x1 + (x2 - x1) * 0.80))))
    top = max(0, min(height, int(np.floor(y1 + (y2 - y1) * 0.18))))
    bottom = max(0, min(height, int(np.ceil(y1 + (y2 - y1) * 0.58))))
    if right - left < 4 or bottom - top < 4:
        return None

    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    saturation_mask = cv2.inRange(hsv, (0, 35, 25), (179, 255, 255))
    chromatic = cv2.calcHist(
        [hsv], [0, 1], saturation_mask, [18, 4], [0, 180, 0, 256]
    ).reshape(-1)
    brightness = cv2.calcHist([hsv], [2], None, [8], [0, 256]).reshape(-1)
    chromatic_total = float(chromatic.sum())
    brightness_total = float(brightness.sum())
    if chromatic_total < 8 or brightness_total < 16:
        return None
    chromatic /= chromatic_total
    brightness /= brightness_total
    feature = np.concatenate((chromatic * 0.75, brightness * 0.25)).astype(
        np.float32
    )
    return feature


def _validate_uniform_input(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("frame debe ser una imagen BGR uint8 con forma (alto, ancho, 3)")
    if not frame.shape[0] or not frame.shape[1]:
        raise ValueError("frame no puede estar vacío")
    try:
        coordinates = np.asarray(box, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("las coordenadas de box deben ser cuatro números finitos") from error
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        raise ValueError("las coordenadas de box deben ser cuatro números finitos")
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        raise ValueError("box debe tener ancho y alto positivos")
    return coordinates


def _mean_histogram(samples: list[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.stack(samples), axis=0)
    total = float(mean.sum())
    if total <= 0:
        raise ValueError("El histograma de apariencia del uniforme está vacío")
    return (mean / total).astype(np.float32)


def _histogram_distance(left: np.ndarray, right: np.ndarray) -> float:
    coefficient = float(np.sqrt(left * right).sum())
    return float(np.sqrt(max(0.0, 1.0 - min(coefficient, 1.0))))


def _two_cluster_kmeans(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.zeros((len(features), len(features)), dtype=np.float64)
    for left in range(len(features)):
        for right in range(left + 1, len(features)):
            distance = _histogram_distance(features[left], features[right])
            distances[left, right] = distance
            distances[right, left] = distance
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    if first == second:
        return np.zeros(len(features), dtype=np.int32), np.stack(
            (features[first], features[second])
        )
    centers = np.stack((features[first], features[second])).astype(np.float32)
    labels = np.zeros(len(features), dtype=np.int32)
    for _ in range(20):
        squared = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        next_labels = np.argmin(squared, axis=1).astype(np.int32)
        if np.array_equal(next_labels, labels) and _ > 0:
            break
        labels = next_labels
        if any(np.count_nonzero(labels == cluster) == 0 for cluster in (0, 1)):
            break
        updated = np.stack(
            [features[labels == cluster].mean(axis=0) for cluster in (0, 1)]
        )
        totals = updated.sum(axis=1, keepdims=True)
        centers = updated / np.maximum(totals, 1e-12)
    return labels, centers
