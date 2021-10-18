import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_mosaic.models.mosaic_config import MosaicConfig
from photo_mosaic.models.raw_image import (
    RAW_IMAGE_CURRENT_JPEG,
    RAW_IMAGE_CURRENT_SMALL_JPEG,
    RAW_IMAGE_ORIGINAL_JPEG,
)
from photo_mosaic.utils.image_processing import bytes2pil, np2pil, pil2bytes, pil2np

SEGMENT_WIDTH = 30
SEGMENT_HEIGHT = 40
N_ROWS = 15
N_COLS = 15

config = MosaicConfig(
    title="Test Mosaic",
    num_segments=32,
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
        np.ones((SEGMENT_HEIGHT * N_ROWS, SEGMENT_WIDTH * 5, 3), dtype="uint8") * 50,
        np.ones((SEGMENT_HEIGHT * N_ROWS, SEGMENT_WIDTH * 5, 3), dtype="uint8") * 100,
        np.ones((SEGMENT_HEIGHT * N_ROWS, SEGMENT_WIDTH * 5, 3), dtype="uint8") * 150,
    ]
)

dark_portrait = np.ones((SEGMENT_HEIGHT, SEGMENT_WIDTH, 3), dtype="uint8") * 35
medium_portrait = np.ones((SEGMENT_HEIGHT, SEGMENT_WIDTH, 3), dtype="uint8") * 85
bright_portrait = np.ones((SEGMENT_HEIGHT, SEGMENT_WIDTH, 3), dtype="uint8") * 200


def create_mosaic(client, image, mosaic_config) -> str:
    response = client.post(
        "/mosaic/",
        files={"file": ("filename", pil2bytes(np2pil(image)), "image/jpeg")},
        data={
            "title": mosaic_config.title,
            "num_segments": mosaic_config.num_segments,
            "mosaic_bg_brightness": mosaic_config.mosaic_bg_brightness,
            "mosaic_blend_value": mosaic_config.mosaic_blend_value,
            "segment_blend_value": mosaic_config.segment_blend_value,
            "segment_blur_low": mosaic_config.segment_blur_low,
            "segment_blur_medium": mosaic_config.segment_blur_medium,
            "segment_blur_high": mosaic_config.segment_blur_high,
        },
    )
    assert response.status_code == 200
    return response.headers["mosaic_id"]


@pytest.fixture(scope="function")
def prepare_db(request, tmp_path):
    db_path = tmp_path / "db"
    db_path.mkdir()
    os.environ["SQLITE_PATH"] = str(db_path)

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


def test_mosaic_end(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    mosaic_id = create_mosaic(client, image_0, config)

    # fill mosaic
    for _ in range(10):
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
    assert np.amin(filled_np) >= 42 and np.amax(filled_np) <= 162  # tolerance
    filled_np = pil2np(bytes2pil(db.read_raw_image(mosaic_id, RAW_IMAGE_CURRENT_SMALL_JPEG).image_bytes))
    assert filled_np.shape == (512, 384, 3)
    assert np.amin(filled_np) >= 42 and np.amax(filled_np) <= 162  # tolerance
    original = pil2np(bytes2pil(db.read_raw_image(mosaic_id, RAW_IMAGE_ORIGINAL_JPEG).image_bytes))
    assert original.shape == (512, 384, 3)

    # check if new mosaic created
    mosaic_list = db.read_mosaic_list()
    new_mosaic_id = None
    for m_id, _, _, _, _, original in mosaic_list:
        if not original:
            new_mosaic_id = m_id
    assert new_mosaic_id is not None
    new_metadata = db.read_mosaic_metadata(new_mosaic_id)
    assert new_metadata.filled is False and new_metadata.active is True
    assert (
        new_metadata.n_rows == metadata.n_rows
        and new_metadata.n_cols == metadata.n_cols
        and new_metadata.segment_height == metadata.segment_height
        and new_metadata.segment_width == metadata.segment_width
        and new_metadata.mosaic_config.title == metadata.mosaic_config.title
    )


def test_mosaic_filled_next_becomes_active(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    mosaic_id_0 = create_mosaic(client, image_0, config)
    mosaic_id_1 = create_mosaic(client, image_0, config)
    mosaic_id_2 = create_mosaic(client, image_0, config)

    # fill mosaic
    for _ in range(10):
        response = client.post(
            f"/mosaic/{mosaic_id_0}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200

    # check if mosaic is closed
    metadata_0 = db.read_mosaic_metadata(mosaic_id_0)
    assert metadata_0.filled is True and metadata_0.active is False

    # check if next mosaic is active
    metadata_1 = db.read_mosaic_metadata(mosaic_id_1)
    assert metadata_1.filled is False and metadata_1.active is True

    # check if next+1 mosaic is not active
    metadata_2 = db.read_mosaic_metadata(mosaic_id_2)
    assert metadata_2.filled is False and metadata_2.active is False


def test_mosaic_delete_next_becomes_active(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    mosaic_id_0 = create_mosaic(client, image_0, config)
    mosaic_id_1 = create_mosaic(client, image_0, config)
    mosaic_id_2 = create_mosaic(client, image_0, config)

    response = client.delete(f"/mosaic/{mosaic_id_0}")
    assert response.status_code == 200

    # check if next mosaic is active
    metadata_1 = db.read_mosaic_metadata(mosaic_id_1)
    assert metadata_1.filled is False and metadata_1.active is True

    # check if next+1 mosaic is not active
    metadata_2 = db.read_mosaic_metadata(mosaic_id_2)
    assert metadata_2.filled is False and metadata_2.active is False


def test_mosaic_delete_filled(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    mosaic_id_0 = create_mosaic(client, image_0, config)
    mosaic_id_1 = create_mosaic(client, image_0, config)
    mosaic_id_2 = create_mosaic(client, image_0, config)

    # fill mosaic
    for _ in range(10):
        response = client.post(
            f"/mosaic/{mosaic_id_0}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200
    response = client.delete(f"/mosaic/{mosaic_id_0}")
    assert response.status_code == 200

    # check if next mosaic is active
    metadata_1 = db.read_mosaic_metadata(mosaic_id_1)
    assert metadata_1.filled is False
    assert metadata_1.active is True

    # check if next+1 mosaic is not active
    metadata_2 = db.read_mosaic_metadata(mosaic_id_2)
    assert metadata_2.filled is False
    assert metadata_2.active is False


def test_mosaic_delete_next(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    mosaic_id_0 = create_mosaic(client, image_0, config)
    mosaic_id_1 = create_mosaic(client, image_0, config)

    # fill mosaic
    for _ in range(10):
        response = client.post(
            f"/mosaic/{mosaic_id_0}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200

    response = client.delete(f"/mosaic/{mosaic_id_1}")
    assert response.status_code == 200

    # check if filled is not active
    metadata_0 = db.read_mosaic_metadata(mosaic_id_0)
    assert metadata_0.filled is True
    assert metadata_0.active is False


def test_mosaic_delete_next_2(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    mosaic_id_0 = create_mosaic(client, image_0, config)
    mosaic_id_1 = create_mosaic(client, image_0, config)
    mosaic_id_2 = create_mosaic(client, image_0, config)

    # fill mosaic
    for _ in range(10):
        response = client.post(
            f"/mosaic/{mosaic_id_0}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200

    response = client.delete(f"/mosaic/{mosaic_id_1}")
    assert response.status_code == 200

    # check if filled is not active
    metadata_0 = db.read_mosaic_metadata(mosaic_id_0)
    assert metadata_0.filled is True
    assert metadata_0.active is False

    # check if next+1 mosaic is active
    metadata_2 = db.read_mosaic_metadata(mosaic_id_2)
    assert metadata_2.filled is False
    assert metadata_2.active is True


def test_mosaic_delete_next_3(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    mosaic_id_0 = create_mosaic(client, image_0, config)
    mosaic_id_1 = create_mosaic(client, image_0, config)
    mosaic_id_2 = create_mosaic(client, image_0, config)

    # fill mosaic
    for _ in range(10):
        response = client.post(
            f"/mosaic/{mosaic_id_0}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200

    response = client.delete(f"/mosaic/{mosaic_id_2}")
    assert response.status_code == 200

    # check if filled is not active
    metadata_0 = db.read_mosaic_metadata(mosaic_id_0)
    assert metadata_0.filled is True
    assert metadata_0.active is False

    # check if next+1 mosaic is active
    metadata_1 = db.read_mosaic_metadata(mosaic_id_1)
    assert metadata_1.filled is False
    assert metadata_1.active is True


def test_mosaic_delete_last(prepare_db):
    # pylint: disable=redefined-outer-name
    client, _ = prepare_db
    mosaic_id_0 = create_mosaic(client, image_0, config)

    response = client.delete(f"/mosaic/{mosaic_id_0}")
    assert response.status_code == 200


def test_mosaic_fill_delete_add_fill_delete_delete_fill(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    config_2 = config.copy()
    config_2.title = "Test2"
    config_3 = config.copy()
    config_3.title = "Test3"
    mosaic_id_0 = create_mosaic(client, image_0, config)
    mosaic_id_1 = create_mosaic(client, image_0, config_2)
    mosaic_id_2 = create_mosaic(client, image_0, config_3)

    # fill mosaic
    for _ in range(10):
        response = client.post(
            f"/mosaic/{mosaic_id_0}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200

    response = client.delete(f"/mosaic/{mosaic_id_1}")
    assert response.status_code == 200

    # fill mosaic
    for _ in range(10):
        response = client.post(
            f"/mosaic/{mosaic_id_2}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200

    # check if filled is not active
    metadata_0 = db.read_mosaic_metadata(mosaic_id_0)
    assert metadata_0.filled is True
    assert metadata_0.active is False

    # check if next+1 mosaic is active
    metadata_2 = db.read_mosaic_metadata(mosaic_id_2)
    assert metadata_2.filled is True
    assert metadata_2.active is False

    # check if new mosaics created correctly
    mosaics = db.read_mosaic_list()
    metadata_3 = db.read_mosaic_metadata(mosaics[2][0])
    metadata_4 = db.read_mosaic_metadata(mosaics[3][0])
    assert metadata_3.active is True
    assert metadata_3.original is False
    assert metadata_4.active is False
    assert metadata_4.original is False
    assert metadata_0.mosaic_config.title == metadata_3.mosaic_config.title
    assert metadata_2.mosaic_config.title == metadata_4.mosaic_config.title

    # delete mosaic 2
    response = client.delete(f"/mosaic/{mosaic_id_2}")
    metadata_0 = db.read_mosaic_metadata(metadata_0.id)
    metadata_3 = db.read_mosaic_metadata(metadata_3.id)
    metadata_4 = db.read_mosaic_metadata(metadata_4.id)
    assert response.status_code == 200
    assert metadata_0.active is False
    assert metadata_3.active is True
    assert metadata_4.active is False

    # delete mosaic 3
    response = client.delete(f"/mosaic/{metadata_3.id}")
    metadata_0 = db.read_mosaic_metadata(metadata_0.id)
    metadata_4 = db.read_mosaic_metadata(metadata_4.id)
    assert response.status_code == 200
    assert metadata_0.active is False
    assert metadata_4.active is True

    # fill mosaic 4
    for _ in range(10):
        response = client.post(
            f"/mosaic/{metadata_4.id}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200

    # check if new mosaic is created correctly
    mosaics = db.read_mosaic_list()
    metadata_0 = db.read_mosaic_metadata(mosaics[0][0])
    metadata_4 = db.read_mosaic_metadata(mosaics[1][0])
    metadata_5 = db.read_mosaic_metadata(mosaics[2][0])
    assert metadata_0.active is False
    assert metadata_4.active is False
    assert metadata_5.active is True
    assert metadata_5.original is False
    assert metadata_0.mosaic_config.title == metadata_5.mosaic_config.title


def test_mosaic_delete_last_original(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    mosaic_id_0 = create_mosaic(client, image_0, config)

    # fill mosaic 0
    for _ in range(10):
        response = client.post(
            f"/mosaic/{mosaic_id_0}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200

    response = client.delete(f"/mosaic/{mosaic_id_0}")
    assert response.status_code == 200

    # check if filled is not active
    mosaics = db.read_mosaic_list()
    metadata_1 = db.read_mosaic_metadata(mosaics[0][0])
    assert metadata_1.filled is False
    assert metadata_1.active is True

    # fill mosaic 1
    for _ in range(10):
        response = client.post(
            f"/mosaic/{metadata_1.id}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200

    # check if no other mosaics are created
    mosaics = db.read_mosaic_list()
    assert len(mosaics) == 1
    metadata_1 = db.read_mosaic_metadata(metadata_1.id)
    assert metadata_1.filled is True
    assert metadata_1.active is False


def test_delete_multiple_actives(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    mosaic_id_0 = create_mosaic(client, image_0, config)
    mosaic_id_1 = create_mosaic(client, image_0, config)
    mosaic_id_2 = create_mosaic(client, image_0, config)
    mosaic_id_3 = create_mosaic(client, image_0, config)
    meta_0 = db.read_mosaic_metadata(mosaic_id_0)
    meta_0.active = True
    meta_0.filled = False
    meta_0.original = True
    db.update_mosaic_metadata(meta_0)
    meta_1 = db.read_mosaic_metadata(mosaic_id_1)
    meta_1.active = True
    meta_1.filled = False
    meta_1.original = True
    db.update_mosaic_metadata(meta_1)
    meta_2 = db.read_mosaic_metadata(mosaic_id_2)
    meta_2.active = True
    meta_2.filled = False
    meta_2.original = False
    db.update_mosaic_metadata(meta_2)
    meta_3 = db.read_mosaic_metadata(mosaic_id_3)
    meta_3.active = True
    meta_3.filled = False
    meta_3.original = False
    db.update_mosaic_metadata(meta_3)

    # delete mosaic 1
    response = client.delete(f"/mosaic/{mosaic_id_1}")
    assert response.status_code == 200

    mosaics = db.read_mosaic_list()
    assert mosaics[0][3] == 1
    assert mosaics[0][5] == 1
    assert mosaics[1][3] == 0
    assert mosaics[1][5] == 0
    assert mosaics[2][3] == 0
    assert mosaics[2][5] == 0


def test_fill_multiple_actives(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    mosaic_id_0 = create_mosaic(client, image_0, config)
    mosaic_id_1 = create_mosaic(client, image_0, config)
    mosaic_id_2 = create_mosaic(client, image_0, config)
    mosaic_id_3 = create_mosaic(client, image_0, config)
    meta_0 = db.read_mosaic_metadata(mosaic_id_0)
    meta_0.active = True
    meta_0.filled = False
    meta_0.original = True
    db.update_mosaic_metadata(meta_0)
    meta_1 = db.read_mosaic_metadata(mosaic_id_1)
    meta_1.active = True
    meta_1.filled = False
    meta_1.original = True
    db.update_mosaic_metadata(meta_1)
    meta_2 = db.read_mosaic_metadata(mosaic_id_2)
    meta_2.active = True
    meta_2.filled = False
    meta_2.original = False
    db.update_mosaic_metadata(meta_2)
    meta_3 = db.read_mosaic_metadata(mosaic_id_3)
    meta_3.active = True
    meta_3.filled = False
    meta_3.original = False
    db.update_mosaic_metadata(meta_3)

    # fill mosaic 2
    for _ in range(10):
        response = client.post(
            f"/mosaic/{meta_2.id}/segment",
            files={"file": ("filename", pil2bytes(np2pil(bright_portrait)), "image/jpeg")},
        )
        assert response.status_code == 200

    mosaics = db.read_mosaic_list()
    assert mosaics[0][3] == 1
    assert mosaics[0][5] == 1
    assert mosaics[1][3] == 0
    assert mosaics[1][5] == 1
    assert mosaics[2][3] == 0
    assert mosaics[2][5] == 0
    assert mosaics[3][3] == 0
    assert mosaics[3][5] == 0


def test_correct_multiple_actives(prepare_db):
    # pylint: disable=redefined-outer-name
    client, db = prepare_db
    mosaic_id_0 = create_mosaic(client, image_0, config)
    mosaic_id_1 = create_mosaic(client, image_0, config)
    mosaic_id_2 = create_mosaic(client, image_0, config)
    mosaic_id_3 = create_mosaic(client, image_0, config)
    meta_0 = db.read_mosaic_metadata(mosaic_id_0)
    meta_0.active = True
    meta_0.filled = False
    meta_0.original = True
    db.update_mosaic_metadata(meta_0)
    meta_1 = db.read_mosaic_metadata(mosaic_id_1)
    meta_1.active = True
    meta_1.filled = False
    meta_1.original = True
    db.update_mosaic_metadata(meta_1)
    meta_2 = db.read_mosaic_metadata(mosaic_id_2)
    meta_2.active = True
    meta_2.filled = False
    meta_2.original = False
    db.update_mosaic_metadata(meta_2)
    meta_3 = db.read_mosaic_metadata(mosaic_id_3)
    meta_3.active = True
    meta_3.filled = False
    meta_3.original = False
    db.update_mosaic_metadata(meta_3)

    response = client.post(
        f"/mosaic/{mosaic_id_1}/states",
        data={"active": False, "filled": False, "original": True},
    )
    assert response.status_code == 200

    response = client.post(
        f"/mosaic/{mosaic_id_2}/states",
        data={"active": False, "filled": False, "original": False},
    )
    assert response.status_code == 200

    response = client.post(
        f"/mosaic/{mosaic_id_3}/states",
        data={"active": False, "filled": False, "original": False},
    )
    assert response.status_code == 200

    mosaics = db.read_mosaic_list()
    assert mosaics[0][3] == 1
    assert mosaics[0][5] == 1
    assert mosaics[1][3] == 0
    assert mosaics[1][5] == 1
    assert mosaics[2][3] == 0
    assert mosaics[2][5] == 0
    assert mosaics[3][3] == 0
    assert mosaics[3][5] == 0
