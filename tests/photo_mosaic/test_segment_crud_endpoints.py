import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_mosaic.models.mosaic_config import MosaicConfig
from photo_mosaic.models.mosaic_metadata import MosaicMetadata
from photo_mosaic.models.raw_image import (
    RAW_IMAGE_CURRENT_JPEG,
    RAW_IMAGE_CURRENT_SMALL_JPEG,
    RAW_IMAGE_FILLING_GIF,
    RAW_IMAGE_ORIGINAL_JPEG,
    RawImage,
)
from photo_mosaic.models.segment import Segment
from photo_mosaic.utils.image_processing import np2pil, pil2bytes

SEGMENT_WIDTH = 30
SEGMENT_HEIGHT = 40
N_ROWS = N_COLS = 10

config = MosaicConfig(
    title="Test Mosaic",
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
    active=True,
    filled=False,
    original=True,
    segment_width=SEGMENT_WIDTH,
    segment_height=SEGMENT_HEIGHT,
    n_rows=N_ROWS,
    n_cols=N_COLS,
    space_top=1,
    space_left=1,
    mosaic_config=config,
)

s_0 = Segment(
    id="11111111-2222-3333-4444-555555555551",
    mosaic_id=m_0.id,
    row_idx=5,
    col_idx=6,
    x_min=m_0.space_left + 6 * m_0.segment_width,
    x_max=m_0.space_left + (6 + 1) * m_0.segment_width,
    y_min=m_0.space_top + 5 * m_0.segment_height,
    y_max=m_0.space_top + (5 + 1) * m_0.segment_height,
    brightness=0,
    fillable=True,
    filled=False,
    is_start_segment=True,
    random_sort_key=0,
)

s_1 = Segment(
    id="11111111-2222-3333-4444-555555555552",
    mosaic_id=m_0.id,
    row_idx=8,
    col_idx=7,
    x_min=m_0.space_left + 8 * m_0.segment_width,
    x_max=m_0.space_left + (8 + 1) * m_0.segment_width,
    y_min=m_0.space_top + 7 * m_0.segment_height,
    y_max=m_0.space_top + (7 + 1) * m_0.segment_height,
    brightness=1,
    fillable=True,
    filled=False,
    is_start_segment=False,
    random_sort_key=1,
)

s_2 = Segment(
    id="11111111-2222-3333-4444-555555555553",
    mosaic_id=m_0.id,
    row_idx=0,
    col_idx=0,
    x_min=m_0.space_left + 0 * m_0.segment_width,
    x_max=m_0.space_left + (0 + 1) * m_0.segment_width,
    y_min=m_0.space_top + 0 * m_0.segment_height,
    y_max=m_0.space_top + (0 + 1) * m_0.segment_height,
    brightness=2,
    fillable=False,
    filled=True,
    is_start_segment=True,
    random_sort_key=2,
)

s_3 = Segment(
    id="11111111-2222-3333-4444-555555555554",
    mosaic_id=m_0.id,
    row_idx=9,
    col_idx=9,
    x_min=m_0.space_left + 9 * m_0.segment_width,
    x_max=m_0.space_left + (9 + 1) * m_0.segment_width,
    y_min=m_0.space_top + 9 * m_0.segment_height,
    y_max=m_0.space_top + (9 + 1) * m_0.segment_height,
    brightness=2,
    fillable=False,
    filled=True,
    is_start_segment=False,
    random_sort_key=3,
)

image_0 = np.zeros((SEGMENT_HEIGHT * N_ROWS, SEGMENT_WIDTH * N_COLS, 3), dtype="uint8")
image_0_bytes = pil2bytes(np2pil(image_0))
cur_jpeg_0 = RawImage(mosaic_id=m_0.id, category=RAW_IMAGE_CURRENT_JPEG, image_bytes=image_0_bytes)
cur_jpeg_small_0 = RawImage(mosaic_id=m_0.id, category=RAW_IMAGE_CURRENT_SMALL_JPEG, image_bytes=image_0_bytes)
orig_jpeg_0 = RawImage(mosaic_id=m_0.id, category=RAW_IMAGE_ORIGINAL_JPEG, image_bytes=image_0_bytes)
filling_gif_0 = RawImage(mosaic_id=m_0.id, category=RAW_IMAGE_FILLING_GIF, image_bytes=image_0_bytes)


@pytest.fixture(scope="function")
def prepare_db(request, tmp_path):
    db_path = tmp_path / "db"
    db_path.mkdir()
    os.environ["SQLITE_PATH"] = str(db_path)

    from photo_mosaic.services.persistence import db

    db.connect()
    db.insert_mosaic_metadata(m_0)
    db.upsert_raw_image(cur_jpeg_0)
    db.upsert_raw_image(cur_jpeg_small_0)
    db.upsert_raw_image(orig_jpeg_0)
    db.upsert_raw_image(filling_gif_0)
    db.upsert_segments([s_0, s_1, s_2, s_3])
    db.commit()

    from photo_mosaic.app import app

    client = TestClient(app)

    def teardown_db():
        db.delete_mosaic_metadata(m_0.id)
        db.commit()
        db.disconnect()

    request.addfinalizer(teardown_db)
    return client, db


def test_list_segments(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    db.connect()
    db.insert_mosaic_metadata(m_0)
    db.upsert_segments([s_0, s_1, s_2, s_3])
    db.commit()

    response = client.get(f"/mosaic/{m_0.id}/segment/list")
    assert response.status_code == 200
    expected_res = {
        "segment_list": [
            {
                "id": "11111111-2222-3333-4444-555555555551",
                "idx": 56,
                "row": 5,
                "col": 6,
                "bri": 0,
                "fillable": 1,
                "filled": 0,
                "start": 1,
            },
            {
                "id": "11111111-2222-3333-4444-555555555552",
                "idx": 87,
                "row": 8,
                "col": 7,
                "bri": 1,
                "fillable": 1,
                "filled": 0,
                "start": 0,
            },
            {
                "id": "11111111-2222-3333-4444-555555555553",
                "idx": 0,
                "row": 0,
                "col": 0,
                "bri": 2,
                "fillable": 0,
                "filled": 1,
                "start": 1,
            },
            {
                "id": "11111111-2222-3333-4444-555555555554",
                "idx": 99,
                "row": 9,
                "col": 9,
                "bri": 2,
                "fillable": 0,
                "filled": 1,
                "start": 0,
            },
        ]
    }
    assert response.json() == expected_res
