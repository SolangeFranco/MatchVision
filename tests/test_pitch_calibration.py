import pytest

from matchvision.pitch.calibration import (
    CalibrationError,
    fit_pitch_calibration,
)


def _calibration_points() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    image = [(100, 100), (500, 100), (500, 400), (100, 400)]
    pitch = [(0, 0), (40, 0), (40, 30), (0, 30)]
    return image, pitch


def test_fit_and_project_points_with_explicit_nonphysical_scale() -> None:
    image, pitch = _calibration_points()
    calibration = fit_pitch_calibration(
        image,
        pitch,
        image_size=(640, 480),
        units="unidades_de_campo",
        scale_verified=False,
        ransac_reprojection_threshold=0.01,
        reference_frame_index=12,
    )

    projected = calibration.project_points([(300, 250)])

    assert projected[0] == pytest.approx((20, 15))
    assert calibration.inlier_count == 4
    assert calibration.reprojection_rmse < 1e-8
    assert calibration.reference_frame_index == 12
    assert calibration.to_dict()["scale_verified"] is False
    assert "no está verificada" in calibration.to_dict()["limitations"][2]


def test_fit_uses_ransac_to_reject_an_inconsistent_correspondence() -> None:
    image = [
        (100, 100),
        (300, 100),
        (500, 100),
        (100, 250),
        (300, 250),
        (500, 250),
        (100, 400),
        (300, 400),
        (500, 400),
    ]
    pitch = [(x / 10 - 10, y / 10 - 10) for x, y in image]
    pitch[-1] = (70, -50)

    calibration = fit_pitch_calibration(
        image,
        pitch,
        image_size=(640, 480),
        units="m",
        scale_verified=True,
        ransac_reprojection_threshold=0.05,
        maximum_reprojection_rmse=0.01,
    )

    assert calibration.inlier_count == 8
    assert calibration.inlier_mask[-1] is False
    assert calibration.reprojection_rmse < 1e-8


def test_project_player_footpoint_uses_bottom_center_of_box() -> None:
    image, pitch = _calibration_points()
    calibration = fit_pitch_calibration(
        image,
        pitch,
        image_size=(640, 480),
        units="m",
        scale_verified=False,
        ransac_reprojection_threshold=0.01,
    )

    assert calibration.project_player_footpoint((280, 180, 320, 250)) == pytest.approx(
        (20, 15)
    )
    assert calibration.project_player_footpoint((280, 400, 320, 480))[1] == pytest.approx(
        38
    )


@pytest.mark.parametrize(
    ("image_points", "pitch_points", "message"),
    [
        ([(1, 1), (2, 2), (3, 3)], [(1, 1), (2, 2), (3, 3)], "al menos cuatro"),
        (
            [(1, 1), (2, 2), (3, 3), (4, 4)],
            [(1, 1), (2, 2), (3, 3)],
            "Cada punto",
        ),
        (
            [(1, 1), (2, 2), (3, 3), (4, 4)],
            [(1, 1), (2, 2), (3, 4), (4, 5)],
            "singular",
        ),
        (
            [(1, 1), (2, 2), (3, 3), (640, 4)],
            [(1, 1), (2, 2), (3, 3), (4, 4)],
            "dentro de la imagen",
        ),
        (
            [(1, 1), (2, 2), (float("nan"), 3), (4, 4)],
            [(1, 1), (2, 2), (3, 3), (4, 4)],
            "no finitas",
        ),
    ],
)
def test_fit_rejects_invalid_correspondences(
    image_points: list[tuple[float, float]],
    pitch_points: list[tuple[float, float]],
    message: str,
) -> None:
    with pytest.raises(CalibrationError, match=message):
        fit_pitch_calibration(
            image_points,
            pitch_points,
            image_size=(640, 480),
            units="m",
            scale_verified=True,
            ransac_reprojection_threshold=0.1,
        )


def test_fit_rejects_high_reprojection_error_when_a_limit_is_requested() -> None:
    image = [(100, 100), (500, 100), (500, 400), (100, 400), (300, 250)]
    pitch = [(0, 0), (40, 0), (40, 30), (0, 30), (24, 17)]

    with pytest.raises(CalibrationError, match="RMSE de reproyección"):
        fit_pitch_calibration(
            image,
            pitch,
            image_size=(640, 480),
            units="m",
            scale_verified=False,
            ransac_reprojection_threshold=100,
            maximum_reprojection_rmse=0.01,
        )


def test_fit_validates_frame_size_units_and_projection_points() -> None:
    image, pitch = _calibration_points()
    options = {
        "image_size": (640, 480),
        "units": "m",
        "scale_verified": False,
        "ransac_reprojection_threshold": 0.01,
    }
    with pytest.raises(CalibrationError, match="image_size"):
        fit_pitch_calibration(image, pitch, **{**options, "image_size": (0, 480)})
    with pytest.raises(CalibrationError, match="units"):
        fit_pitch_calibration(image, pitch, **{**options, "units": "  "})

    calibration = fit_pitch_calibration(image, pitch, **options)
    with pytest.raises(CalibrationError, match="fuera"):
        calibration.project_points([(640, 250)])
    with pytest.raises(CalibrationError, match="área positiva"):
        calibration.project_player_footpoint((2, 2, 2, 4))
