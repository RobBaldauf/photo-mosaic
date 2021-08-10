import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_mosaic.models.image_pixels import (
    IMAGE_PIXELS_CATEGORY_CURRENT,
    IMAGE_PIXELS_CATEGORY_ORIGINAL,
)
from photo_mosaic.models.mosaic_config import MosaicConfig
from photo_mosaic.models.mosaic_metadata import MosaicMetadata
from photo_mosaic.models.raw_image import (
    RAW_IMAGE_CURRENT_JPEG,
    RAW_IMAGE_CURRENT_SMALL_JPEG,
    RAW_IMAGE_FILLING_GIF,
    RAW_IMAGE_ORIGINAL_JPEG,
)
from photo_mosaic.utils.image_processing import bytes2pil, np2pil, pil2bytes, pil2np

SEGMENT_WIDTH = 30
SEGMENT_HEIGHT = 40
N_ROWS = N_COLS = 20

config = MosaicConfig(
    num_segments=200,
    mosaic_bg_brightness=0.2,
    mosaic_blend_value=0.2,
    segment_blend_value=0.3,
    segment_blur_low=3,
    segment_blur_medium=2,
    segment_blur_high=1,
)

image_0 = np.ones((SEGMENT_HEIGHT * N_ROWS, SEGMENT_WIDTH * N_COLS, 3), dtype="uint8") * 127
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


def test_create_mosaic(prepare_db):
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

    # check metadata
    mosaic_id = response.headers["mosaic_id"]
    metadata = MosaicMetadata(
        id=mosaic_id,
        idx=1,
        active=True,
        filled=False,
        original=True,
        segment_width=42,
        segment_height=56,
        n_rows=14,
        n_cols=14,
        space_top=8,
        space_left=6,
        mosaic_config=config,
    )
    res_metadata = db.read_mosaic_metadata(mosaic_id)
    assert metadata == res_metadata

    # check np arrays
    orig_pixels = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_ORIGINAL)
    np.testing.assert_array_equal(image_0, orig_pixels.pixel_array)

    cur_pixels = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_CURRENT)
    np.testing.assert_array_equal(image_0_reduced_brightness, cur_pixels.pixel_array)

    # check raw images
    orig_np = pil2np(bytes2pil(db.read_raw_image(mosaic_id, RAW_IMAGE_ORIGINAL_JPEG).image_bytes))
    np.testing.assert_array_equal(image_0_small_orig, orig_np)

    cur_np = pil2np(bytes2pil(db.read_raw_image(mosaic_id, RAW_IMAGE_CURRENT_JPEG).image_bytes))
    np.testing.assert_array_equal(image_0_reduced_brightness, cur_np)

    cur_small_np = pil2np(bytes2pil(db.read_raw_image(mosaic_id, RAW_IMAGE_CURRENT_SMALL_JPEG).image_bytes))
    np.testing.assert_array_equal(image_0_small_reduced_brightness, cur_small_np)

    gif_bytes = db.read_raw_image(mosaic_id, RAW_IMAGE_FILLING_GIF).image_bytes
    assert gif_bytes is not None

    # check segments
    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 0, 1: 0, 2: 196}
    assert stats == expected_stats

    fillable_segments = db.get_segments(mosaic_id=mosaic_id, fillable=True)
    assert len(fillable_segments) == 10


def test_delete_mosaic(prepare_db):
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

    response = client.delete(f"/mosaic/{mosaic_id}")
    assert response.status_code == 200
    assert not db.mosaic_exists(mosaic_id)


def test_reset_mosaic(prepare_db):
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
        f"/mosaic/{mosaic_id}/segment",
        files={"file": ("filename", pil2bytes(np2pil(image_0)), "image/jpeg")},
        data={"quick_fill": "false"},
    )
    assert response.status_code == 200

    response = client.post(f"/mosaic/{mosaic_id}/reset")
    assert response.status_code == 200

    # check metadata
    metadata = MosaicMetadata(
        id=mosaic_id,
        idx=1,
        active=True,
        filled=False,
        original=True,
        segment_width=42,
        segment_height=56,
        n_rows=14,
        n_cols=14,
        space_top=8,
        space_left=6,
        mosaic_config=config,
    )
    res_metadata = db.read_mosaic_metadata(mosaic_id)
    assert metadata == res_metadata

    # check np arrays
    orig_pixels = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_ORIGINAL)
    np.testing.assert_array_equal(image_0, orig_pixels.pixel_array)

    cur_pixels = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_CURRENT)
    np.testing.assert_array_equal(image_0_reduced_brightness, cur_pixels.pixel_array)

    # check raw images
    orig_np = pil2np(bytes2pil(db.read_raw_image(mosaic_id, RAW_IMAGE_ORIGINAL_JPEG).image_bytes))
    np.testing.assert_array_equal(image_0_small_orig, orig_np)

    cur_np = pil2np(bytes2pil(db.read_raw_image(mosaic_id, RAW_IMAGE_CURRENT_JPEG).image_bytes))
    np.testing.assert_array_equal(image_0_reduced_brightness, cur_np)

    cur_small_np = pil2np(bytes2pil(db.read_raw_image(mosaic_id, RAW_IMAGE_CURRENT_SMALL_JPEG).image_bytes))
    np.testing.assert_array_equal(image_0_small_reduced_brightness, cur_small_np)

    gif_bytes = db.read_raw_image(mosaic_id, RAW_IMAGE_FILLING_GIF).image_bytes
    assert gif_bytes is not None

    # check segments
    stats = db.get_segment_stats(mosaic_id)
    expected_stats = {0: 0, 1: 0, 2: 196}
    assert stats == expected_stats

    fillable_segments = db.get_segments(mosaic_id=mosaic_id, fillable=True)
    assert len(fillable_segments) == 10


def test_create_mosaic_invalid_param(prepare_db):
    # pylint: disable=redefined-outer-name
    client, _ = prepare_db
    response = client.post(
        "/mosaic/",
        files={"file": ("filename", pil2bytes(np2pil(image_0)), "image/jpeg")},
        data={
            "num_segments": config.num_segments,
            "mosaic_bg_brightness": -1,
            "mosaic_blend_value": config.mosaic_blend_value,
            "segment_blend_value": config.segment_blend_value,
            "segment_blur_low": config.segment_blur_low,
            "segment_blur_medium": config.segment_blur_medium,
            "segment_blur_high": config.segment_blur_high,
        },
    )
    assert response.status_code == 422
