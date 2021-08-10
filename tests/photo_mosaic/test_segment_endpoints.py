import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_mosaic.models.image_pixels import IMAGE_PIXELS_CATEGORY_CURRENT
from photo_mosaic.models.mosaic_config import MosaicConfig
from photo_mosaic.models.raw_image import (
    RAW_IMAGE_CURRENT_JPEG,
    RAW_IMAGE_CURRENT_SMALL_JPEG,
)
from photo_mosaic.utils.image_processing import bytes2pil, np2pil, pil2bytes, pil2np

SEGMENT_WIDTH = 30
SEGMENT_HEIGHT = 40
N_ROWS = 10
N_COLS = 3

config = MosaicConfig(
    num_segments=30,
    mosaic_bg_brightness=0.2,
    mosaic_blend_value=0.2,
    segment_blend_value=0.3,
    segment_blur_low=3,
    segment_blur_medium=2,
    segment_blur_high=1,
)

# image with three vertical stripes of different brightness
image_0 = np.hstack(
    [
        np.ones((SEGMENT_HEIGHT * N_ROWS, SEGMENT_WIDTH, 3), dtype="uint8") * 50,
        np.ones((SEGMENT_HEIGHT * N_ROWS, SEGMENT_WIDTH, 3), dtype="uint8") * 100,
        np.ones((SEGMENT_HEIGHT * N_ROWS, SEGMENT_WIDTH, 3), dtype="uint8") * 150,
    ]
)

dark_portrait = np.ones((SEGMENT_HEIGHT, SEGMENT_WIDTH, 3), dtype="uint8") * 35
medium_portrait = np.ones((SEGMENT_HEIGHT, SEGMENT_WIDTH, 3), dtype="uint8") * 85
bright_portrait = np.ones((SEGMENT_HEIGHT, SEGMENT_WIDTH, 3), dtype="uint8") * 200

image_0_reduced_brightness = np.ones((SEGMENT_HEIGHT * N_ROWS, SEGMENT_WIDTH * N_COLS, 3), dtype="uint8") * 25
image_0_small_orig = np.ones((512, 384, 3), dtype="uint8") * 127
image_0_small_reduced_brightness = np.ones((512, 384, 3), dtype="uint8") * 25


@pytest.fixture(scope="function")
def prepare_db(request, tmp_path):
    db_path = tmp_path / "db"
    db_path.mkdir()
    os.environ["SQL_LITE_PATH"] = str(db_path)

    from photo_mosaic.app import app
    from photo_mosaic.services.persistence import db

    client = TestClient(app)

    def teardown_db():
        db.disconnect()
        mosaics = db.read_mosaic_list()
        for m in mosaics:
            db.delete_mosaic_metadata(m[0])

    request.addfinalizer(teardown_db)
    return client, db


def test_segment_brightness_detection(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    response = client.post(
        "/mosaic/",
        files={"file": ("filename", pil2bytes(np2pil(image_0)), "image/jpeg")},
        data={
            "num_segments": config.num_segments,
            "mosaic_bg_brightness": config.mosaic_bg_brightness,
            "mosaic_blend_value": config.mosaic_blend_value,
            "segment_blend_value": config.segment_blend_value,
            "segment_blur_low": config.segment_blur_low,
            "segment_blur_medium": config.segment_blur_medium,
            "segment_blur_high": config.segment_blur_high,
        },
    )
    assert response.status_code == 200
    mosaic_id = response.headers["mosaic_id"]

    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 10, 1: 10, 2: 10}
    assert stats == expected_stats

    fillable_segments = db.get_segments(mosaic_id=mosaic_id, fillable=True)
    assert len(fillable_segments) == 30


def test_segment_sampling(prepare_db):
    # pylint: disable=redefined-outer-name
    client, _ = prepare_db
    response = client.post(
        "/mosaic/",
        files={"file": ("filename", pil2bytes(np2pil(image_0)), "image/jpeg")},
        data={
            "num_segments": config.num_segments,
            "mosaic_bg_brightness": config.mosaic_bg_brightness,
            "mosaic_blend_value": config.mosaic_blend_value,
            "segment_blend_value": config.segment_blend_value,
            "segment_blur_low": config.segment_blur_low,
            "segment_blur_medium": config.segment_blur_medium,
            "segment_blur_high": config.segment_blur_high,
        },
    )
    assert response.status_code == 200
    mosaic_id = response.headers["mosaic_id"]

    # test dark sampling
    response = client.post(
        f"/mosaic/{mosaic_id}/segment/sample",
        files={"file": ("filename", pil2bytes(np2pil(dark_portrait)), "image/jpeg")},
    )
    assert response.status_code == 200
    img = pil2np(bytes2pil(response.content))
    assert img.shape == dark_portrait.shape
    assert np.amin(img) >= 45 and np.amax(img) <= 47  # tolerance

    # test medium sampling
    response = client.post(
        f"/mosaic/{mosaic_id}/segment/sample",
        files={"file": ("filename", pil2bytes(np2pil(medium_portrait)), "image/jpeg")},
    )
    assert response.status_code == 200
    img = pil2np(bytes2pil(response.content))
    assert img.shape == dark_portrait.shape
    assert np.amin(img) >= 95 and np.amax(img) <= 97

    # test bright sampling
    response = client.post(
        f"/mosaic/{mosaic_id}/segment/sample",
        files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
    )
    assert response.status_code == 200
    img = pil2np(bytes2pil(response.content))
    assert img.shape == dark_portrait.shape
    assert np.amin(img) >= 165 and np.amax(img) <= 167


def test_segment_filling(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    response = client.post(
        "/mosaic/",
        files={"file": ("filename", pil2bytes(np2pil(image_0)), "image/jpeg")},
        data={
            "num_segments": config.num_segments,
            "mosaic_bg_brightness": config.mosaic_bg_brightness,
            "mosaic_blend_value": config.mosaic_blend_value,
            "segment_blend_value": config.segment_blend_value,
            "segment_blur_low": config.segment_blur_low,
            "segment_blur_medium": config.segment_blur_medium,
            "segment_blur_high": config.segment_blur_high,
        },
    )
    assert response.status_code == 200
    mosaic_id = response.headers["mosaic_id"]

    response = client.post(
        f"/mosaic/{mosaic_id}/segment/sample",
        files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
    )
    assert response.status_code == 200
    segment_id = response.headers["segment_id"]

    response = client.post(
        f"/mosaic/{mosaic_id}/segment/{segment_id}",
        files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
    )
    assert response.status_code == 200

    seg = db.get_segments(mosaic_id=mosaic_id, id=segment_id)[0]
    assert seg.filled is True and seg.fillable is False and seg.is_start_segment is True

    cur_pixels = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_CURRENT).pixel_array
    seg_pixels = cur_pixels[seg.y_min : seg.y_max, seg.x_min : seg.x_max]
    assert np.amin(seg_pixels) >= 159 and np.amax(seg_pixels) <= 161  # jpeg compression tolerance

    # check raw images
    cur_np = pil2np(bytes2pil(db.read_raw_image(mosaic_id, RAW_IMAGE_CURRENT_JPEG).image_bytes))
    seg_pixels = cur_np[seg.y_min : seg.y_max, seg.x_min : seg.x_max]
    assert np.amin(seg_pixels) >= 158 and np.amax(seg_pixels) <= 162  # jpeg compression tolerance


def test_segment_filling_order_dark(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    response = client.post(
        "/mosaic/",
        files={"file": ("filename", pil2bytes(np2pil(image_0)), "image/jpeg")},
        data={
            "num_segments": config.num_segments,
            "mosaic_bg_brightness": config.mosaic_bg_brightness,
            "mosaic_blend_value": config.mosaic_blend_value,
            "segment_blend_value": config.segment_blend_value,
            "segment_blur_low": config.segment_blur_low,
            "segment_blur_medium": config.segment_blur_medium,
            "segment_blur_high": config.segment_blur_high,
        },
    )
    assert response.status_code == 200
    mosaic_id = response.headers["mosaic_id"]

    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(dark_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(dark_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 0, 1: 10, 2: 10}
    assert stats == expected_stats
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(dark_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(dark_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 0, 1: 0, 2: 10}
    assert stats == expected_stats
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(dark_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200

    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 0, 1: 0, 2: 5}
    assert stats == expected_stats


def test_segment_filling_order_medium(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    response = client.post(
        "/mosaic/",
        files={"file": ("filename", pil2bytes(np2pil(image_0)), "image/jpeg")},
        data={
            "num_segments": config.num_segments,
            "mosaic_bg_brightness": config.mosaic_bg_brightness,
            "mosaic_blend_value": config.mosaic_blend_value,
            "segment_blend_value": config.segment_blend_value,
            "segment_blur_low": config.segment_blur_low,
            "segment_blur_medium": config.segment_blur_medium,
            "segment_blur_high": config.segment_blur_high,
        },
    )
    assert response.status_code == 200
    mosaic_id = response.headers["mosaic_id"]

    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(medium_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(medium_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 10, 1: 0, 2: 10}
    assert stats == expected_stats
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(medium_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(medium_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 0, 1: 0, 2: 10}
    assert stats == expected_stats
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(medium_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200

    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 0, 1: 0, 2: 5}
    assert stats == expected_stats


def test_segment_filling_order_bright(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    response = client.post(
        "/mosaic/",
        files={"file": ("filename", pil2bytes(np2pil(image_0)), "image/jpeg")},
        data={
            "num_segments": config.num_segments,
            "mosaic_bg_brightness": config.mosaic_bg_brightness,
            "mosaic_blend_value": config.mosaic_blend_value,
            "segment_blend_value": config.segment_blend_value,
            "segment_blur_low": config.segment_blur_low,
            "segment_blur_medium": config.segment_blur_medium,
            "segment_blur_high": config.segment_blur_high,
        },
    )
    assert response.status_code == 200
    mosaic_id = response.headers["mosaic_id"]

    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 10, 1: 10, 2: 0}
    assert stats == expected_stats
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200
    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 10, 1: 0, 2: 0}
    assert stats == expected_stats
    response = client.post(
        f"/mosaic/{mosaic_id}/segment", files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")}
    )
    assert response.status_code == 200

    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 5, 1: 0, 2: 0}
    assert stats == expected_stats


def test_segment_filling_mosaic_end(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    response = client.post(
        "/mosaic/",
        files={"file": ("filename", pil2bytes(np2pil(image_0)), "image/jpeg")},
        data={
            "num_segments": config.num_segments,
            "mosaic_bg_brightness": config.mosaic_bg_brightness,
            "mosaic_blend_value": config.mosaic_blend_value,
            "segment_blend_value": config.segment_blend_value,
            "segment_blur_low": config.segment_blur_low,
            "segment_blur_medium": config.segment_blur_medium,
            "segment_blur_high": config.segment_blur_high,
        },
    )
    assert response.status_code == 200
    mosaic_id = response.headers["mosaic_id"]

    # fill mosaic
    for _ in range(4):
        response = client.post(
            f"/mosaic/{mosaic_id}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200
    response = client.post(
        f"/mosaic/{mosaic_id}/segment",
        files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        data={"quick_fill": "false"},
    )
    assert response.status_code == 200

    # check if mosaic is closed
    metadata = db.read_mosaic_metadata(mosaic_id)
    assert metadata.filled is True and metadata.active is False
    filled_np = pil2np(bytes2pil(db.read_raw_image(mosaic_id, RAW_IMAGE_CURRENT_JPEG).image_bytes))
    assert filled_np.shape == image_0.shape
    assert np.amin(filled_np) >= 45 and np.amax(filled_np) <= 162  # tolerance
    filled_np = pil2np(bytes2pil(db.read_raw_image(mosaic_id, RAW_IMAGE_CURRENT_SMALL_JPEG).image_bytes))
    assert filled_np.shape == image_0.shape
    assert np.amin(filled_np) >= 45 and np.amax(filled_np) <= 162  # tolerance

    # check if new mosaic created
    mosaic_list = db.read_mosaic_list()
    new_mosaic_id = None
    for m_id, _, _, _, original in mosaic_list:
        if not original:
            new_mosaic_id = m_id
    assert new_mosaic_id is not None
    metadata = db.read_mosaic_metadata(new_mosaic_id)
    assert metadata.filled is False and metadata.active is True
