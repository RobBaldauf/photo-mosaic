import atexit
import io
import os
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
from fastapi import HTTPException

from photo_mosaic.models.image_pixels import ImagePixels
from photo_mosaic.models.mosaic_config import MosaicConfig
from photo_mosaic.models.mosaic_metadata import MosaicMetadata
from photo_mosaic.models.raw_image import RawImage
from photo_mosaic.models.segment import Segment
from photo_mosaic.services.abstract_persistence import AbstractPersistenceService

MOSAIC_METADATA_TABLE = "mosaic_metadata"
SEGMENT_TABLE = "segments"
RAW_IMAGE_TABLE = "raw_images"
IMAGE_PIXELS_TABLE = "image_pixels"


def _write_np_array(array):
    byte_arr = io.BytesIO()
    np.save(byte_arr, array)
    byte_arr.seek(0)
    return sqlite3.Binary(byte_arr.read())


def _read_np_array(blob):
    byte_arr = io.BytesIO(blob)
    byte_arr.seek(0)
    return np.load(byte_arr)


sqlite3.register_adapter(np.ndarray, _write_np_array)
sqlite3.register_converter("array", _read_np_array)


class SQLiteAbstractPersistenceService(AbstractPersistenceService):
    def __init__(self, path: str):
        if not os.path.isdir(path):
            raise ValueError(f"SQL_LITE_PATH {path} is not a directory!")

        self._path = os.path.join(path, "mosaic.db")
        self._connection = None
        atexit.register(self._destroy)

    def _destroy(self):
        self.disconnect()

    def _connect(self):
        if not self._connection:
            if not os.path.isfile(self._path):
                self._init_db()
            self._connection = sqlite3.connect(self._path, detect_types=sqlite3.PARSE_DECLTYPES)
        return self._connection

    def commit(self):
        if self._connection:
            self._connection.commit()

    def disconnect(self):
        if self._connection:
            self._connection.close()
            self._connection = None

    def mosaic_exists(self, mosaic_id: str) -> bool:
        con = self._connect()
        cur = con.cursor()
        cur.execute(f"""SELECT EXISTS(SELECT 1 FROM {MOSAIC_METADATA_TABLE} WHERE id=?);""", (mosaic_id,))
        return bool(cur.fetchone())

    def mosaic_count(self) -> int:
        con = self._connect()
        cur = con.cursor()
        cur.execute(f"""SELECT COUNT(*) FROM {MOSAIC_METADATA_TABLE};""")
        num = cur.fetchone()[0]
        return num

    def read_mosaic_list(self) -> List[Tuple[str, int, bool, bool, bool]]:
        con = self._connect()
        cur = con.cursor()
        query = f"""SELECT id, idx, active, filled, original FROM {MOSAIC_METADATA_TABLE};"""
        cur.execute(query)
        rows = cur.fetchall()
        return rows

    def insert_mosaic_metadata(self, metadata: MosaicMetadata):
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            f"""INSERT OR REPLACE INTO {MOSAIC_METADATA_TABLE} (id, active, filled,
               original, segment_width, segment_height, n_rows, n_cols, space_top, space_left, num_segments,
               mosaic_background_brightness, mosaic_blend_value, segment_blend_value, segment_blur_low,
               segment_blur_medium, segment_blur_high) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                metadata.id,
                metadata.active,
                metadata.filled,
                metadata.original,
                metadata.segment_width,
                metadata.segment_height,
                metadata.n_rows,
                metadata.n_cols,
                metadata.space_top,
                metadata.space_left,
                metadata.mosaic_config.num_segments,
                metadata.mosaic_config.mosaic_background_brightness,
                metadata.mosaic_config.mosaic_blend_value,
                metadata.mosaic_config.segment_blend_value,
                metadata.mosaic_config.segment_blur_low,
                metadata.mosaic_config.segment_blur_medium,
                metadata.mosaic_config.segment_blur_high,
            ),
        )

    def update_mosaic_metadata(self, metadata: MosaicMetadata):
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            f"""INSERT OR REPLACE INTO {MOSAIC_METADATA_TABLE} (id, idx, active, filled,
            original, segment_width, segment_height, n_rows, n_cols, space_top, space_left, num_segments,
            mosaic_background_brightness, mosaic_blend_value, segment_blend_value, segment_blur_low,
            segment_blur_medium, segment_blur_high) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                metadata.id,
                metadata.idx,
                metadata.active,
                metadata.filled,
                metadata.original,
                metadata.segment_width,
                metadata.segment_height,
                metadata.n_rows,
                metadata.n_cols,
                metadata.space_top,
                metadata.space_left,
                metadata.mosaic_config.num_segments,
                metadata.mosaic_config.mosaic_background_brightness,
                metadata.mosaic_config.mosaic_blend_value,
                metadata.mosaic_config.segment_blend_value,
                metadata.mosaic_config.segment_blur_low,
                metadata.mosaic_config.segment_blur_medium,
                metadata.mosaic_config.segment_blur_high,
            ),
        )

    def read_mosaic_metadata(self, mosaic_id: str, active_only: bool = False) -> MosaicMetadata:
        con = self._connect()
        cur = con.cursor()
        query = f"""SELECT num_segments, mosaic_background_brightness, mosaic_blend_value, segment_blend_value,
                segment_blur_low, segment_blur_medium, segment_blur_high, idx, active, filled, original, segment_width,
                segment_height, n_rows, n_cols, space_top, space_left FROM {MOSAIC_METADATA_TABLE} WHERE id=?"""
        if active_only:
            query += " AND active=1;"
        else:
            query += ";"
            cur.execute(
                query,
                (mosaic_id,),
            )
        rows = cur.fetchall()
        if len(rows) == 0:
            raise HTTPException(status_code=404, detail=f"Mosaic {mosaic_id} does not exist.")
        row = rows[0]
        config = MosaicConfig(
            num_segments=row[0],
            mosaic_background_brightness=row[1],
            mosaic_blend_value=row[2],
            segment_blend_value=row[3],
            segment_blur_low=row[4],
            segment_blur_medium=row[5],
            segment_blur_high=row[6],
        )
        return MosaicMetadata(
            id=mosaic_id,
            idx=row[7],
            active=row[8],
            filled=row[9],
            original=row[10],
            segment_width=row[11],
            segment_height=row[12],
            n_rows=row[13],
            n_cols=row[14],
            space_top=row[15],
            space_left=row[16],
            mosaic_config=config,
        )

    def delete_mosaic_metadata(self, mosaic_id: str):
        mosaic_list = self.read_mosaic_list()
        new_active = None
        exists = False
        for m_id, _, _, _, active in mosaic_list:
            if exists:
                new_active = m_id
                break
            if m_id == mosaic_id:
                exists = True
                if not active:
                    break
            else:
                new_active = m_id

        if exists:
            con = self._connect()
            cur = con.cursor()
            cur.execute(f"""DELETE FROM {MOSAIC_METADATA_TABLE} WHERE id=?;""", (mosaic_id,))
            if new_active:
                cur.execute(f"""UPDATE {MOSAIC_METADATA_TABLE} SET active = 1  WHERE id = ?;""", (new_active,))
        else:
            raise HTTPException(status_code=404, detail=f"Mosaic {mosaic_id} does not exist.")

    def upsert_raw_image(self, raw_image: RawImage):
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            f"""INSERT OR REPLACE INTO {RAW_IMAGE_TABLE} (mosaic_id, category, image_bytes)
            values (?, ?, ?)""",
            (
                raw_image.mosaic_id,
                raw_image.category,
                sqlite3.Binary(raw_image.image_bytes),
            ),
        )

    def read_raw_image(self, mosaic_id: str, category: int) -> RawImage:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            f"""SELECT image_bytes FROM {RAW_IMAGE_TABLE}
        WHERE mosaic_id=? AND category=?;""",
            (
                mosaic_id,
                category,
            ),
        )
        rows = cur.fetchall()

        if len(rows) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No raw image exist for mosaic_id={mosaic_id} AND category={category} does not exist.",
            )
        row = rows[0]
        return RawImage(mosaic_id=mosaic_id, category=category, image_bytes=row[0])

    def upsert_image_pixels(self, image_pixels: ImagePixels):
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            f"""INSERT OR REPLACE INTO {IMAGE_PIXELS_TABLE} (mosaic_id, category, pixel_array)
            values (?, ?, ?)""",
            (
                image_pixels.mosaic_id,
                image_pixels.category,
                image_pixels.pixel_array,
            ),
        )

    def read_image_pixels(self, mosaic_id: str, category: int) -> ImagePixels:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            f"""SELECT pixel_array FROM {IMAGE_PIXELS_TABLE} WHERE mosaic_id=? AND category=?;""",
            (
                mosaic_id,
                category,
            ),
        )
        rows = cur.fetchall()

        if len(rows) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No raw image exist for mosaic_id={mosaic_id} " f"AND category={category} does not exist.",
            )
        row = rows[0]
        return ImagePixels(mosaic_id=mosaic_id, category=category, pixel_array=row[0])

    def segment_exists(self, segment_id: str) -> bool:
        con = self._connect()
        cur = con.cursor()
        cur.execute(f"""SELECT EXISTS(SELECT 1 FROM {SEGMENT_TABLE} WHERE id=?);""", (segment_id,))
        return bool(cur.fetchone())

    def upsert_segment(self, seg: Segment):
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            f"""INSERT OR REPLACE INTO {SEGMENT_TABLE} (id, mosaic_id, row_idx, col_idx, x_min, x_max, y_min, y_max,
                        brightness, fillable, filled, is_start_segment) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                seg.id,
                seg.mosaic_id,
                seg.row_idx,
                seg.col_idx,
                seg.x_min,
                seg.x_max,
                seg.y_min,
                seg.y_max,
                seg.brightness,
                int(seg.fillable),
                int(seg.filled),
                int(seg.is_start_segment),
            ),
        )

    def upsert_segments(self, segments: List[Segment]):
        con = self._connect()
        cur = con.cursor()
        for seg in segments:
            cur.execute(
                f"""INSERT OR REPLACE INTO {SEGMENT_TABLE} (id, mosaic_id, row_idx, col_idx, x_min, x_max, y_min, y_max,
                brightness, fillable, filled, is_start_segment) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    seg.id,
                    seg.mosaic_id,
                    seg.row_idx,
                    seg.col_idx,
                    seg.x_min,
                    seg.x_max,
                    seg.y_min,
                    seg.y_max,
                    seg.brightness,
                    int(seg.fillable),
                    int(seg.filled),
                    int(seg.is_start_segment),
                ),
            )

    def get_segment_stats(self, mosaic_id: str) -> Dict[int, int]:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            f"""SELECT brightness, COUNT(id) FROM {SEGMENT_TABLE}
            WHERE mosaic_id=? AND filled=0 GROUP BY brightness;""",
            (mosaic_id,),
        )
        rows = cur.fetchall()

        res_dict = {r[0]: r[1] for r in rows}
        for brightness in [0, 1, 2]:
            if brightness not in res_dict:
                res_dict[brightness] = 0
        return res_dict

    def get_segments(self, limit: int = -1, offset: int = 0, random_order: bool = False, **kwargs) -> List[Segment]:
        keys = [
            "id",
            "mosaic_id",
            "row_idx",
            "col_idx",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "brightness",
            "fillable",
            "filled",
            "is_start_segment",
        ]

        where_keys = []
        where_values = []
        for k, v in kwargs.items():
            if k in keys:
                if isinstance(v, list):
                    where_keys.append(f"{k} IN ?")
                    where_values.append(v)
                else:
                    where_keys.append(f"{k}=?")
                    where_values.append(v)
        query = f"SELECT {', '.join(keys)} FROM {SEGMENT_TABLE} WHERE {' AND '.join(where_keys)}"
        if random_order:
            query += " ORDER BY RANDOM()"
        if limit > 0:
            query += f" LIMIT {limit}"
            query += f" OFFSET {offset};"
        con = self._connect()
        cur = con.cursor()
        cur.execute(query, where_values)
        rows = cur.fetchall()

        segments = []
        for row in rows:
            segments.append(
                Segment(
                    id=row[0],
                    mosaic_id=row[1],
                    row_idx=row[2],
                    col_idx=row[3],
                    x_min=row[4],
                    x_max=row[5],
                    y_min=row[6],
                    y_max=row[7],
                    brightness=row[8],
                    fillable=row[9],
                    filled=row[10],
                    is_start_segment=row[11],
                )
            )
        return segments

    def _init_db(self):
        con = sqlite3.connect(self._path, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        cur.execute(
            f"""CREATE TABLE {MOSAIC_METADATA_TABLE}
                           (idx INTEGER PRIMARY KEY,
                            id TEXT UNIQUE,
                            active INTEGER,
                            filled INTEGER,
                            original INTEGER,
                            segment_width INTEGER,
                            segment_height INTEGER,
                            n_rows INTEGER,
                            n_cols INTEGER,
                            space_top INTEGER,
                            space_left INTEGER,
                            num_segments INTEGER,
                            mosaic_background_brightness REAL,
                            mosaic_blend_value REAL,
                            segment_blend_value REAL,
                            segment_blur_low  REAL,
                            segment_blur_medium REAL,
                            segment_blur_high REAL
                            )"""
        )

        cur.execute(
            f"""CREATE TABLE {RAW_IMAGE_TABLE}
                           (mosaic_id TEXT,
                            category INTEGER,
                            image_bytes BLOB,
                            PRIMARY KEY (mosaic_id, category),
                            CONSTRAINT fk_mosaic_id
                              FOREIGN KEY (mosaic_id)
                              REFERENCES {MOSAIC_METADATA_TABLE}(id)
                              ON DELETE CASCADE)"""
        )

        cur.execute(
            f"""CREATE TABLE {IMAGE_PIXELS_TABLE}
                           (mosaic_id TEXT,
                            category INTEGER,
                            pixel_array array,
                            PRIMARY KEY (mosaic_id, category),
                            CONSTRAINT fk_mosaic_id
                              FOREIGN KEY (mosaic_id)
                              REFERENCES {MOSAIC_METADATA_TABLE}(id)
                              ON DELETE CASCADE)"""
        )

        cur.execute(
            f"""CREATE TABLE {SEGMENT_TABLE}
                           (id TEXT PRIMARY KEY,
                            mosaic_id TEXT,
                            row_idx INTEGER,
                            col_idx INTEGER,
                            x_min INTEGER,
                            x_max INTEGER,
                            y_min INTEGER,
                            y_max INTEGER,
                            brightness INTEGER,
                            fillable INTEGER,
                            filled INTEGER,
                            is_start_segment INTEGER,
                            CONSTRAINT fk_mosaic_id
                              FOREIGN KEY (mosaic_id)
                              REFERENCES {MOSAIC_METADATA_TABLE}(id)
                              ON DELETE CASCADE
                            )"""
        )
        self.commit()
        self.disconnect()
