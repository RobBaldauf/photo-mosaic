import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_mosaic.models.mosaic_config import MosaicConfig
from photo_mosaic.models.mosaic_metadata import MosaicMetadata
from photo_mosaic.models.raw_image import (
    RAW_IMAGE_CURRENT_JPEG,
    RAW_IMAGE_CURRENT_SMALL_JPEG,
    RawImage,
)
from photo_mosaic.utils.image_processing import bytes2pil, np2pil, pil2bytes, pil2np

SEGMENT_WIDTH = 30
SEGMENT_HEIGHT = 40
N_ROWS = N_COLS = 10

config_0 = MosaicConfig(
    title="Test Mosaic",
    num_segments=100,
    mosaic_bg_brightness=0.25,
    mosaic_blend_value=0.25,
    segment_blend_value=0.4,
    segment_blur_low=4,
    segment_blur_medium=3,
    segment_blur_high=2,
)

config_1 = MosaicConfig(
    title="",
    num_segments=100,
    mosaic_bg_brightness=0.25,
    mosaic_blend_value=0.25,
    segment_blend_value=0.4,
    segment_blur_low=4,
    segment_blur_medium=3,
    segment_blur_high=2,
)

m_0 = MosaicMetadata(
    id="11111111-2222-3333-4444-555555555550",
    idx=-1,
    active=False,
    filled=True,
    original=True,
    segment_width=SEGMENT_WIDTH,
    segment_height=SEGMENT_HEIGHT,
    n_rows=N_ROWS,
    n_cols=N_COLS,
    space_top=1,
    space_left=1,
    mosaic_config=config_0,
)

m_1 = MosaicMetadata(
    id="11111111-2222-3333-4444-555555555551",
    idx=-1,
    active=True,
    filled=False,
    original=False,
    segment_width=SEGMENT_WIDTH,
    segment_height=SEGMENT_HEIGHT,
    n_rows=N_ROWS,
    n_cols=N_COLS,
    space_top=1,
    space_left=1,
    mosaic_config=config_1,
)

image_0 = np.zeros((SEGMENT_HEIGHT * N_ROWS, SEGMENT_WIDTH * N_COLS, 3), dtype="uint8")
image_1 = np.ones((SEGMENT_HEIGHT * N_ROWS, SEGMENT_WIDTH * N_COLS, 3), dtype="uint8")
cur_jpeg_0 = RawImage(mosaic_id=m_0.id, category=RAW_IMAGE_CURRENT_JPEG, image_bytes=pil2bytes(np2pil(image_0)))
cur_jpeg_1 = RawImage(mosaic_id=m_1.id, category=RAW_IMAGE_CURRENT_JPEG, image_bytes=pil2bytes(np2pil(image_1)))
image_small_0 = np2pil(image_0.copy())
image_small_1 = np2pil(image_1.copy())
image_small_0.thumbnail((512, 512))
image_small_1.thumbnail((512, 512))
image_small_0 = pil2np(image_small_0)
image_small_1 = pil2np(image_small_1)
cur_jpeg_small_0 = RawImage(
    mosaic_id=m_0.id, category=RAW_IMAGE_CURRENT_SMALL_JPEG, image_bytes=pil2bytes(np2pil(image_small_0))
)
cur_jpeg_small_1 = RawImage(
    mosaic_id=m_1.id, category=RAW_IMAGE_CURRENT_SMALL_JPEG, image_bytes=pil2bytes(np2pil(image_small_1))
)


@pytest.fixture(scope="function")
def prepare_db(request, tmp_path):
    db_path = tmp_path / "db"
    db_path.mkdir()
    os.environ["SQL_LITE_PATH"] = str(db_path)

    from photo_mosaic.services.persistence import db

    db.connect()
    db.insert_mosaic_metadata(m_0)
    db.insert_mosaic_metadata(m_1)
    db.upsert_raw_image(cur_jpeg_0)
    db.upsert_raw_image(cur_jpeg_1)
    db.upsert_raw_image(cur_jpeg_small_0)
    db.upsert_raw_image(cur_jpeg_small_1)
    db.commit()

    from photo_mosaic.app import app

    client = TestClient(app)

    def teardown_db():
        db.delete_mosaic_metadata(m_0.id)
        db.delete_mosaic_metadata(m_1.id)
        db.commit()
        db.disconnect()

    request.addfinalizer(teardown_db)
    return client


def test_list_all_mosaics(prepare_db):
    # pylint: disable=redefined-outer-name
    response = prepare_db.get("/mosaic/list")
    assert response.status_code == 200
    json_res = {
        "mosaic_list": [
            {"id": m_0.id, "index": 1, "title": config_0.title},
            {"id": m_1.id, "index": 2, "title": config_1.title},
        ]
    }
    assert response.json() == json_res


def test_list_active_mosaics(prepare_db):
    # pylint: disable=redefined-outer-name
    response = prepare_db.get("/mosaic/list/active")
    assert response.status_code == 200
    json_res = {"mosaic_list": [{"id": m_1.id, "index": 2, "title": config_1.title}]}
    assert response.json() == json_res


def test_list_original_mosaics(prepare_db):
    # pylint: disable=redefined-outer-name
    response = prepare_db.get("/mosaic/list/original")
    assert response.status_code == 200
    json_res = {"mosaic_list": [{"id": m_0.id, "index": 1, "title": config_0.title}]}
    assert response.json() == json_res


def test_list_filled_mosaics(prepare_db):
    # pylint: disable=redefined-outer-name
    response = prepare_db.get("/mosaic/list/filled")
    assert response.status_code == 200
    json_res = {"mosaic_list": [{"id": m_0.id, "index": 1, "title": config_0.title}]}
    assert response.json() == json_res


def test_get_current_mosaic(prepare_db):
    # pylint: disable=redefined-outer-name
    response = prepare_db.get(f"/mosaic/{m_0.id}")
    assert response.status_code == 200
    img = pil2np(bytes2pil(response.content))
    np.testing.assert_array_equal(img, image_0)


def test_get_current_mosaic_non_existing_id(prepare_db):
    # pylint: disable=redefined-outer-name
    response = prepare_db.get("/mosaic/11111111-2222-3333-4444-555555555559")
    assert response.status_code == 404
    json_res = {"detail": "mosaic id (11111111-2222-3333-4444-555555555559) does not exist."}
    assert response.json() == json_res


def test_get_current_mosaic_invalid_id(prepare_db):
    # pylint: disable=redefined-outer-name
    response = prepare_db.get("/mosaic/1")
    assert response.status_code == 400
    json_res = {"detail": "mosaic id (1) has to be of format UUID4!"}
    assert response.json() == json_res


def test_get_current_mosaic_thumbnail(prepare_db):
    # pylint: disable=redefined-outer-name
    response = prepare_db.get(f"/mosaic/{m_0.id}/thumbnail")
    assert response.status_code == 200
    img = pil2np(bytes2pil(response.content))
    np.testing.assert_array_equal(img, image_small_0)


def test_get_mosaic_metadata(prepare_db):
    # pylint: disable=redefined-outer-name
    response = prepare_db.get(f"/mosaic/{m_0.id}/metadata")
    assert response.status_code == 200
    json_res = {
        "id": "11111111-2222-3333-4444-555555555550",
        "idx": 1,
        "active": False,
        "filled": True,
        "original": True,
        "segment_width": 30,
        "segment_height": 40,
        "n_rows": 10,
        "n_cols": 10,
        "space_top": 1,
        "space_left": 1,
        "mosaic_config": {
            "title": config_0.title,
            "num_segments": 100,
            "mosaic_bg_brightness": 0.25,
            "mosaic_blend_value": 0.25,
            "segment_blend_value": 0.4,
            "segment_blur_low": 4.0,
            "segment_blur_medium": 3.0,
            "segment_blur_high": 2.0,
        },
        "dark_segments_left": 0,
        "medium_segments_left": 0,
        "bright_segments_left": 0,
    }
    assert response.json() == json_res
