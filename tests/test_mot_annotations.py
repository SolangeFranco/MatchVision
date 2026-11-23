from io import StringIO

import pytest

from matchvision.annotations.mot import read_mot_annotations


def test_reads_soccernet_style_mot_row() -> None:
    annotations = read_mot_annotations(StringIO("1,42,10,20,30,40,1,-1,-1,-1\n"))

    assert len(annotations) == 1
    assert annotations[0].frame == 1
    assert annotations[0].track_id == 42
    assert annotations[0].x == 10
    assert annotations[0].width == 30


def test_reads_nine_column_motchallenge_row() -> None:
    annotations = read_mot_annotations(StringIO("1,42,10,20,30,40,1,1,1\n"))

    assert len(annotations) == 1
    assert annotations[0].track_id == 42


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("1,2,3,4,5,6,1\n", "expected 9 or 10 columns"),
        ("0,2,3,4,5,6,1,-1,-1,-1\n", "frame must be one-based"),
        ("1,2,3,4,0,6,1,-1,-1,-1\n", "dimensions must be positive"),
        ("1,2,nan,4,5,6,1,-1,-1,-1\n", "values must be finite"),
    ],
)
def test_rejects_malformed_rows(row: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_mot_annotations(StringIO(row))
