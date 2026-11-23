import pytest

from matchvision.evaluation.reid_splits import build_grouped_split_manifest


def _record(match_id: str, suffix: str, identities: list[str]) -> dict:
    return {
        "match_id": match_id,
        "video_sha256": suffix * 64,
        "identity_ids": identities,
    }


def test_split_keeps_matches_and_shared_identities_in_one_partition() -> None:
    manifest = build_grouped_split_manifest(
        [
            _record("match-a", "a", ["PLAYER-001"]),
            _record("match-b", "b", ["PLAYER-001", "PLAYER-002"]),
            _record("match-c", "c", ["PLAYER-003"]),
            _record("match-d", "d", ["PLAYER-004"]),
        ],
        seed=7,
    )

    assert manifest["status"] == "ready"
    assignments = {
        item["match_id"]: item["split"] for item in manifest["assignments"]
    }
    assert assignments["match-a"] == assignments["match-b"]
    assert len(set(assignments.values())) == 3


def test_one_match_does_not_claim_independent_train_validation_test() -> None:
    manifest = build_grouped_split_manifest(
        [_record("single-match", "a", ["PLAYER-001", "PLAYER-002"])]
    )

    assert manifest["status"] == "insufficient_independent_groups"
    assert manifest["assignments"][0]["split"] == "unassigned"


def test_split_rejects_reused_video_under_different_match_ids() -> None:
    with pytest.raises(ValueError, match="aparece en más de un partido"):
        build_grouped_split_manifest([
            _record("match-a", "a", ["PLAYER-001"]),
            _record("match-b", "a", ["PLAYER-002"]),
            _record("match-c", "c", ["PLAYER-003"]),
        ])


def test_split_rejects_test_only_identity() -> None:
    with pytest.raises(ValueError, match="TEST-ONLY"):
        build_grouped_split_manifest([
            _record("match-a", "a", ["TEST-ONLY-E2E"]),
            _record("match-b", "b", ["PLAYER-002"]),
            _record("match-c", "c", ["PLAYER-003"]),
        ])


@pytest.mark.parametrize(
    ("records", "status", "split_count"),
    [
        ([], "insufficient_independent_groups", 0),
        ([_record("match-a", "a", ["PLAYER-001"])],
         "insufficient_independent_groups", 1),
        ([
            _record("match-a", "a", ["PLAYER-001"]),
            _record("match-b", "b", ["PLAYER-002"]),
        ], "insufficient_independent_groups", 1),
    ],
)
def test_split_does_not_force_three_splits_with_too_few_groups(
    records: list[dict], status: str, split_count: int
) -> None:
    manifest = build_grouped_split_manifest(records, seed=13)

    assert manifest["status"] == status
    actual_splits = {item["split"] for item in manifest["assignments"]}
    assert len(actual_splits) == split_count


def test_split_round_robin_is_deterministic_for_three_independent_groups() -> None:
    records = [
        _record("match-a", "a", ["PLAYER-001"]),
        _record("match-b", "b", ["PLAYER-002"]),
        _record("match-c", "c", ["PLAYER-003"]),
    ]

    first = build_grouped_split_manifest(records, seed=21)
    second = build_grouped_split_manifest(records, seed=21)
    assignments = {item["match_id"]: item["split"] for item in first["assignments"]}

    assert first == second
    assert set(assignments.values()) == {"train", "validation", "test"}


def test_shared_identities_are_grouped_transitively_and_balanced_by_match_count() -> None:
    records = [
        _record("match-a", "a", ["PLAYER-001"]),
        _record("match-b", "b", ["PLAYER-001", "PLAYER-002"]),
        _record("match-c", "c", ["PLAYER-002"]),
        _record("match-d", "d", ["PLAYER-003"]),
        _record("match-e", "e", ["PLAYER-004"]),
    ]

    manifest = build_grouped_split_manifest(records, seed=4)
    assignments = {item["match_id"]: item["split"] for item in manifest["assignments"]}
    group_split = assignments["match-a"]

    assert assignments["match-b"] == group_split == assignments["match-c"]
    assert len(set(assignments.values())) == 3
    assert max(
        sum(split == value for split in assignments.values())
        for value in ("train", "validation", "test")
    ) <= 3


def test_more_independent_groups_are_distributed_deterministically() -> None:
    records = [
        _record(f"match-{index}", format(index, "x"), [f"PLAYER-{index:03d}"])
        for index in range(1, 8)
    ]

    first = build_grouped_split_manifest(records, seed=10)
    second = build_grouped_split_manifest(records, seed=10)
    split_counts = {
        split: sum(item["split"] == split for item in first["assignments"])
        for split in ("train", "validation", "test")
    }

    assert first == second
    assert split_counts == {"train": 3, "validation": 2, "test": 2}
