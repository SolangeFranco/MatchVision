from pathlib import Path

import cv2
import numpy as np
import pytest

import matchvision.reidentification.appearance as appearance_module
from matchvision.detection.base import Detection
from matchvision.reidentification.osnet import (
    MODEL_FILENAME,
    OSNetEmbeddingExtractor,
    cosine_similarity,
    extract_person_crop,
)
from matchvision.reidentification.appearance import PersistentIdentityAssociator
from matchvision.tracking.base import TrackedObject


def _track(track_id: int, x: int = 20) -> TrackedObject:
    return TrackedObject(
        Detection(x, 10, x + 32, 90, 0.95, 0, "person"),
        track_id,
    )


def _frame() -> np.ndarray:
    frame = np.full((120, 100, 3), 20, dtype=np.uint8)
    cv2.rectangle(frame, (20, 10), (52, 90), (30, 80, 190), -1)
    cv2.rectangle(frame, (28, 20), (44, 58), (210, 220, 230), -1)
    cv2.line(frame, (21, 60), (50, 80), (10, 20, 30), 4)
    return frame


def test_cosine_similarity_validates_and_compares_vectors() -> None:
    assert cosine_similarity(np.array([2.0, 0.0]), np.array([3.0, 0.0])) == 1.0
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    with pytest.raises(ValueError, match="misma dimensión"):
        cosine_similarity(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="embedding nulo"):
        cosine_similarity(np.zeros(2), np.ones(2))


def test_player_crop_clips_to_frame_and_skips_tiny_boxes() -> None:
    frame = np.zeros((40, 50, 3), dtype=np.uint8)

    crop = extract_person_crop(frame, (-10, 0, 25, 30))

    assert crop is not None
    assert crop.shape == (30, 25, 3)
    assert extract_person_crop(frame, (1, 1, 5, 12)) is None


@pytest.mark.parametrize("method", ["embedding", "hybrid"])
def test_embedding_methods_reidentify_with_mocked_model(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    class FakeExtractor:
        configuration = {"model": "fake-osnet", "device": "cpu"}

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def encode(self, crops: list[np.ndarray]) -> list[np.ndarray]:
            return [np.array([1.0, 0.0], dtype=np.float32) for _ in crops]

    monkeypatch.setattr(
        appearance_module, "OSNetEmbeddingExtractor", FakeExtractor
    )
    associator = PersistentIdentityAssociator(
        active_identity_protection=1,
        maximum_reidentification_gap=30,
        reid_method=method,
        embedding_weights_path="weights-placeholder.pth",
    )
    frame = _frame()

    original = associator.update([_track(1)], frame, 0)[0]
    associator.update([], frame, 1)
    associator.update([], frame, 2)
    recovered = associator.update([_track(2, x=21)], frame, 3)[0]

    assert recovered.persistent_id == original.persistent_id
    assert recovered.identity_state == "reidentified"
    assert recovered.association_confidence is not None
    assert associator.configuration["reid_method"] == method


def test_embedding_gallery_keeps_only_configured_recent_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExtractor:
        configuration = {"model": "fake-osnet", "device": "cpu"}

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def encode(self, crops: list[np.ndarray]) -> list[np.ndarray]:
            return [np.array([1.0, 0.0], dtype=np.float32) for _ in crops]

    monkeypatch.setattr(
        appearance_module, "OSNetEmbeddingExtractor", FakeExtractor
    )
    associator = PersistentIdentityAssociator(
        reid_method="embedding",
        embedding_weights_path="weights-placeholder.pth",
        embedding_gallery_size=2,
        embedding_sample_interval=1,
    )

    for frame_index in range(5):
        associator.update([_track(1)], _frame(), frame_index)

    assert len(associator._identities[1].embedding_gallery) == 2


def test_osnet_real_checkpoint_loads_and_returns_unit_embeddings() -> None:
    weights = Path("models") / MODEL_FILENAME
    if not weights.is_file():
        pytest.skip("El checkpoint OSNet no está descargado en este entorno.")

    extractor = OSNetEmbeddingExtractor(weights, device="cpu", batch_size=1)
    embeddings = extractor.encode([_frame(), _frame()])

    assert len(embeddings) == 2
    assert embeddings[0].shape == (512,)
    assert np.linalg.norm(embeddings[0]) == pytest.approx(1.0, abs=1e-5)
    assert cosine_similarity(embeddings[0], embeddings[1]) == pytest.approx(1.0)
