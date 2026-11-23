from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from matchvision.evaluation.identity_annotations import (
    IdentityAnnotationError,
    evaluate_identity_predictions,
    evaluation_frames,
    freeze_identity_annotation_file,
    freeze_identity_annotations,
    load_identity_annotations,
    validate_identity_annotations,
)


def _observation(
    annotation_id: str,
    *,
    identity_id: str | None = "player-1",
    unknown: bool = False,
    ambiguous: bool = False,
    ignore: bool = False,
) -> dict[str, object]:
    return {
        "annotation_id": annotation_id,
        "identity_id": identity_id,
        "x1": 10,
        "y1": 10,
        "x2": 20,
        "y2": 30,
        "unknown": unknown,
        "ambiguous": ambiguous,
        "ignore": ignore,
    }


def _frame(
    frame: int,
    observations: list[dict[str, object]],
    *,
    exhaustive: bool = True,
) -> dict[str, object]:
    return {
        "frame": frame,
        "exhaustive": exhaustive,
        "observations": observations,
    }


def _payload(video_path: Path, frame_records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "video": {
            "path": str(video_path),
            "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
            "frame_count": 4,
            "width": 100,
            "height": 100,
            "fps": 25.0,
        },
        "exhaustive_frames": [
            record["frame"] for record in frame_records if record["exhaustive"]
        ],
        "frames": frame_records,
    }


def _write_mot(path: Path, rows: list[str]) -> Path:
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
    return path


def test_unknown_ambiguous_and_ignored_frames_are_not_evaluation_frames(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture video bytes")
    payload = _payload(
        video,
        [
            _frame(1, [_observation("known")]),
            _frame(2, [_observation("unknown", identity_id=None, unknown=True)]),
            _frame(3, [_observation("ambiguous", identity_id=None, ambiguous=True)]),
            _frame(4, [_observation("ignored", identity_id=None, ignore=True)]),
        ],
    )

    validate_identity_annotations(payload)

    assert evaluation_frames(payload) == [1]


def test_formal_json_schema_accepts_export_shape(tmp_path: Path) -> None:
    import jsonschema

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    known = _observation("known")
    known["entity_type"] = "player"
    payload = _payload(video, [{
        **_frame(1, [known]),
        "events": [{
            "event_id": "appearance-1",
            "identity_id": "player-1",
            "event_type": "appearance",
            "notes": "Anotado por una persona",
        }],
    }, _frame(2, [_observation("unknown", identity_id=None, unknown=True)])])
    schema_path = (
        Path(__file__).parents[1]
        / "src"
        / "matchvision"
        / "schemas"
        / "identity-annotations.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload, schema)


def test_ground_truth_accepts_human_entity_types_and_trajectory_events(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    player = _observation("player")
    player["entity_type"] = "player"
    referee = _observation("referee", identity_id="referee-1")
    referee["entity_type"] = "referee"
    payload = _payload(
        video,
        [
            {
                **_frame(1, [player, referee]),
                "events": [{
                    "event_id": "event-1",
                    "identity_id": "player-1",
                    "event_type": "appearance",
                    "notes": "Entra al área visible",
                }],
            }
        ],
    )

    validate_identity_annotations(payload)


def test_ground_truth_rejects_identity_with_conflicting_entity_types(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    player = _observation("player")
    referee = _observation("referee", identity_id="player-1")
    referee["entity_type"] = "referee"
    payload = _payload(video, [_frame(1, [player, referee])])

    with pytest.raises(IdentityAnnotationError, match="conflicting entity types"):
        validate_identity_annotations(payload)


def test_ground_truth_rejects_same_identity_on_two_objects_in_one_frame(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    payload = _payload(
        video,
        [_frame(1, [_observation("first"), _observation("second")])],
    )

    with pytest.raises(IdentityAnnotationError, match="appears more than once"):
        validate_identity_annotations(payload)


def test_ground_truth_allows_same_annotation_id_across_frames(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    payload = _payload(
        video,
        [
            _frame(1, [_observation("stable-box")]),
            _frame(2, [_observation("stable-box")]),
        ],
    )

    validate_identity_annotations(payload)


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (
            {"event_id": "event-1", "identity_id": "player-1",
             "event_type": "camera_cut"},
            "unsupported trajectory event_type",
        ),
        (
            {"event_id": "event-1", "identity_id": "",
             "event_type": "occlusion_start"},
            "require a human identity_id",
        ),
    ],
)
def test_invalid_trajectory_events_are_rejected(
    tmp_path: Path, event: dict[str, object], message: str
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    payload = _payload(video, [{**_frame(1, []), "events": [event]}])

    with pytest.raises(IdentityAnnotationError, match=message):
        validate_identity_annotations(payload)


def test_frozen_ground_truth_detects_annotation_changes(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    payload = _payload(video, [_frame(1, [_observation("player")])])

    frozen = freeze_identity_annotations(
        payload, frozen_at="2026-11-11T12:00:00+00:00"
    )

    assert "dataset" not in payload
    assert frozen["dataset"]["freeze"]["status"] == "frozen"
    validate_identity_annotations(frozen)

    tampered = json.loads(json.dumps(frozen))
    tampered["frames"][0]["observations"][0]["x1"] = 12
    with pytest.raises(IdentityAnnotationError, match="content SHA-256"):
        validate_identity_annotations(tampered)


def test_already_frozen_ground_truth_cannot_be_frozen_again(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    frozen = freeze_identity_annotations(
        _payload(video, [_frame(1, [_observation("player")])])
    )

    with pytest.raises(IdentityAnnotationError, match="already frozen"):
        freeze_identity_annotations(frozen)


@pytest.mark.parametrize(
    "frame_record",
    [
        _frame(1, [], exhaustive=False),
        _frame(1, [], exhaustive=True),
        _frame(
            1,
            [_observation("unknown", identity_id=None, unknown=True)],
            exhaustive=True,
        ),
    ],
)
def test_freezing_requires_an_exhaustive_frame_with_resolved_identities(
    tmp_path: Path, frame_record: dict[str, object]
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")

    with pytest.raises(IdentityAnnotationError, match="exhaustive frame"):
        freeze_identity_annotations(_payload(video, [frame_record]))


def test_freeze_command_creates_verified_new_file_without_overwrite(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    source = tmp_path / "annotations.json"
    source.write_text(
        json.dumps(_payload(video, [_frame(1, [_observation("player")])])),
        encoding="utf-8",
    )
    output = tmp_path / "frozen" / "test-set.json"

    created = freeze_identity_annotation_file(source, output)

    assert created == output.resolve()
    assert load_identity_annotations(created)["dataset"]["freeze"]["status"] == "frozen"
    with pytest.raises(FileExistsError, match="already exists|ya existe"):
        freeze_identity_annotation_file(source, output)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update({"exhaustive_frames": [True]}),
            "exhaustive_frames must be a list",
        ),
        (
            lambda payload: payload["frames"][0]["observations"][0].update(
                {"ambiguous": True, "unknown": True, "identity_id": None}
            ),
            "both unknown and ambiguous",
        ),
        (
            lambda payload: payload["frames"][0]["observations"][0].update(
                {"ignore": "yes"}
            ),
            "ignore must be boolean",
        ),
        (
            lambda payload: payload["frames"][0].update({"exhaustive": False}),
            "is not marked exhaustive",
        ),
    ],
)
def test_invalid_annotation_state_is_rejected(
    tmp_path: Path, mutate: object, message: str
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    payload = _payload(video, [_frame(1, [_observation("known")])])
    mutate(payload)  # type: ignore[operator]

    with pytest.raises(IdentityAnnotationError, match=message):
        validate_identity_annotations(payload)


def test_trackeval_metrics_use_identical_exhaustive_frames_and_verified_video(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"same video bytes")
    records = [
        _frame(1, [_observation("first")]),
        _frame(2, [_observation("unreviewed")], exhaustive=False),
        _frame(3, [_observation("last")]),
    ]
    annotation_path = tmp_path / "identity-ground-truth.json"
    annotation_path.write_text(
        json.dumps(_payload(video, records)), encoding="utf-8"
    )
    baseline = _write_mot(
        tmp_path / "baseline.txt",
        [
            "1,7,10,10,10,20,1,-1,-1,-1",
            "2,7,10,10,10,20,1,-1,-1,-1",
            "3,7,10,10,10,20,1,-1,-1,-1",
        ],
    )
    reid = _write_mot(
        tmp_path / "reid.txt",
        [
            "1,2,10,10,10,20,1,-1,-1,-1",
            "2,2,10,10,10,20,1,-1,-1,-1",
            "3,2,10,10,10,20,1,-1,-1,-1",
        ],
    )

    report = evaluate_identity_predictions(
        annotation_path,
        {"baseline": baseline, "reid": reid},
        video_path=video,
    )

    assert report["eligible_exhaustive_frames"] == [1, 3]
    assert report["video_sha256_verified"] is True
    assert report["annotation_sha256"]
    for result in report["results"].values():
        assert result["evaluated_frames"] == 2
        assert result["HOTA"] == pytest.approx(1.0)
        assert result["MOTA"] == pytest.approx(1.0)
        assert result["IDF1"] == pytest.approx(1.0)
        assert result["ID_Precision"] == pytest.approx(1.0)
        assert result["ID_Recall"] == pytest.approx(1.0)
        assert result["ID_switches"] == 0
        assert result["Fragmentation"] == 0


def test_video_hash_mismatch_and_unsafe_prediction_name_fail(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"original")
    payload = _payload(video, [_frame(1, [_observation("person")])])
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    prediction = _write_mot(
        tmp_path / "prediction.txt", ["1,1,10,10,10,20,1,-1,-1,-1"]
    )

    video.write_bytes(b"changed")
    with pytest.raises(IdentityAnnotationError, match="SHA-256"):
        evaluate_identity_predictions(
            annotation_path, {"baseline": prediction}, video_path=video
        )
    with pytest.raises(ValueError, match="Prediction names"):
        evaluate_identity_predictions(
            annotation_path, {"../escape": prediction}, video_path=video
        )


def test_trackeval_reports_identity_switch_and_identity_precision_recall(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture video")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            _payload(
                video,
                [
                    _frame(1, [_observation("first")]),
                    _frame(3, [_observation("last")]),
                ],
            )
        ),
        encoding="utf-8",
    )
    prediction = _write_mot(
        tmp_path / "switched.txt",
        [
            "1,7,10,10,10,20,1,-1,-1,-1",
            "3,8,10,10,10,20,1,-1,-1,-1",
        ],
    )

    report = evaluate_identity_predictions(
        annotations, {"switched": prediction}, video_path=video
    )
    metrics = report["results"]["switched"]

    assert metrics["ID_switches"] == 1
    assert metrics["ID_Precision"] == pytest.approx(0.5)
    assert metrics["ID_Recall"] == pytest.approx(0.5)
    assert metrics["IDF1"] == pytest.approx(0.5)


def test_no_eligible_frames_is_not_mistaken_for_identity_metrics(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    payload = _payload(
        video,
        [
            _frame(
                1,
                [_observation("ambiguous", identity_id=None, ambiguous=True)],
            )
        ],
    )
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    prediction = _write_mot(
        tmp_path / "prediction.txt", ["1,1,10,10,10,20,1,-1,-1,-1"]
    )

    with pytest.raises(IdentityAnnotationError, match="No eligible exhaustive frames"):
        evaluate_identity_predictions(
            annotation_path, {"baseline": prediction}, video_path=video
        )
