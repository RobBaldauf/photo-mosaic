import json
import random
import uuid
from typing import List, Tuple

from fastapi import HTTPException
from PIL.Image import Image

from photo_mosaic.models.app_config import get_config
from photo_mosaic.models.image_pixels import (
    IMAGE_PIXELS_CATEGORY_CURRENT,
    IMAGE_PIXELS_CATEGORY_ORIGINAL,
    ImagePixels,
)
from photo_mosaic.models.mosaic_config import MosaicConfig
from photo_mosaic.models.mosaic_metadata import MosaicMetadata
from photo_mosaic.models.raw_image import (
    RAW_IMAGE_CURRENT_JPEG,
    RAW_IMAGE_FILLING_GIF,
    RAW_IMAGE_ORIGINAL_JPEG,
    RawImage,
)
from photo_mosaic.models.segment import Segment
from photo_mosaic.services.abstract_persistence import AbstractPersistenceService
from photo_mosaic.utils.image_processing import (
    HIGH_BRIGHTNESS,
    LOW_BRIGHTNESS,
    MEDIUM_BRIGHTNESS,
    adapt_brightness,
    bytes2pil,
    generate_id,
    get_brightness_category,
    get_segment_config,
    mosaic_2_gif,
    np2pil,
    pil2bytes,
    pil2np,
)


class MosaicManagementService:
    """Service for creation, modification, deletion and retrieval of mosaic data"""

    def __init__(self, persistence_service: AbstractPersistenceService):
        self.persistence = persistence_service

    def create_mosaic(self, image_bytes: bytes, config: MosaicConfig) -> str:
        """
        Create a new mosaic image (including its metadata, np pixel_arrays, segements and binary image files)
        :param image_bytes: The binary image to create a mosaic from
        :param config: The configuration for the mosaic creation process
        :return: The mosaic id (UUID)
        """
        # create metadata + original image
        metadata, image = self._create_mosaic_metadata(image_bytes, config)
        self.persistence.insert_mosaic_metadata(metadata)

        # create original jpeg
        original = image.copy()
        original.thumbnail((get_config().original_image_max_size, get_config().original_image_max_size))
        original_jpeg = RawImage(
            mosaic_id=metadata.id, category=RAW_IMAGE_ORIGINAL_JPEG, image_bytes=pil2bytes(original)
        )
        self.persistence.upsert_raw_image(original_jpeg)

        # create original image pixels (np)
        original_pixels = ImagePixels(
            mosaic_id=metadata.id, category=IMAGE_PIXELS_CATEGORY_ORIGINAL, pixel_array=pil2np(image)
        )
        self.persistence.upsert_image_pixels(original_pixels)

        # create current jpeg
        bg_pil_image = adapt_brightness(image, config.mosaic_background_brightness)
        current_jpeg = RawImage(
            mosaic_id=metadata.id, category=RAW_IMAGE_CURRENT_JPEG, image_bytes=pil2bytes(bg_pil_image)
        )
        self.persistence.upsert_raw_image(current_jpeg)

        # create current image pixels (np)
        current_pixels = ImagePixels(
            mosaic_id=metadata.id, category=IMAGE_PIXELS_CATEGORY_CURRENT, pixel_array=pil2np(bg_pil_image)
        )
        self.persistence.upsert_image_pixels(current_pixels)

        # create segments
        segments = self._create_segments(metadata, original_pixels)
        self.persistence.upsert_segments(segments)

        # Commit changes to db
        self.persistence.commit()

        # create filling gif animation
        gif_image = mosaic_2_gif(persistence=self.persistence, mosaic_id=metadata.id)
        self.persistence.upsert_raw_image(gif_image)
        self.persistence.commit()
        self.persistence.disconnect()
        return metadata.id

    def _create_mosaic_metadata(self, image_bytes: bytes, mosaic_cfg: MosaicConfig) -> Tuple[MosaicMetadata, Image]:
        # if no other mosaic exists set this to active
        is_active = self.persistence.mosaic_count() == 0

        # calculate metadata
        image = bytes2pil(image_bytes)
        width, height = image.size
        segment_ratio = (get_config().segment_ratio_width, get_config().segment_ratio_height)
        seg_cfg = get_segment_config(width, height, mosaic_cfg.num_segments, segment_ratio)
        space_top = int((height - seg_cfg["seg_h"] * seg_cfg["num_rows"]) / 2)
        space_left = int((width - seg_cfg["seg_w"] * seg_cfg["num_cols"]) / 2)

        metadata = MosaicMetadata(
            id=generate_id(),
            idx=-1,
            active=is_active,
            filled=False,
            original=True,
            segment_width=seg_cfg["seg_w"],
            segment_height=seg_cfg["seg_h"],
            n_rows=seg_cfg["num_rows"],
            n_cols=seg_cfg["num_cols"],
            space_top=space_top,
            space_left=space_left,
            mosaic_config=mosaic_cfg,
        )
        return metadata, image

    def _create_segments(self, metadata: MosaicMetadata, pixels: ImagePixels) -> List[Segment]:
        # calculate the center of the image
        row_min, col_min, row_max, col_max = self._get_image_center(metadata)
        center = "center"
        edge = "edge"
        segments: dict = {center: {0: [], 1: [], 2: []}, edge: {0: [], 1: [], 2: []}}

        # create segment data structures + determine brightness/location
        for c in range(metadata.n_cols):
            for r in range(metadata.n_rows):
                new_seg = Segment(
                    id=str(uuid.uuid4()),
                    mosaic_id=metadata.id,
                    row_idx=r,
                    col_idx=c,
                    x_min=metadata.space_left + c * metadata.segment_width,
                    x_max=metadata.space_left + (c + 1) * metadata.segment_width,
                    y_min=metadata.space_top + r * metadata.segment_height,
                    y_max=metadata.space_top + (r + 1) * metadata.segment_height,
                    brightness=-1,
                    fillable=False,
                    filled=False,
                    is_start_segment=False,
                )
                np_segment = pixels.pixel_array[new_seg.y_min : new_seg.y_max, new_seg.x_min : new_seg.x_max]
                new_seg.brightness = get_brightness_category(np2pil(np_segment))

                if row_min <= r <= row_max and col_min <= c <= col_max:
                    position = center
                else:
                    position = edge
                if new_seg.brightness not in [LOW_BRIGHTNESS, MEDIUM_BRIGHTNESS, HIGH_BRIGHTNESS]:
                    raise HTTPException(status_code=400, detail="Invalid image data")
                segments[position][new_seg.brightness].append(new_seg)

        # randomly select n segments for each brightness to be fillable (image center is prefered)
        num_segments_start = get_config().num_segments_start
        for brightness in [LOW_BRIGHTNESS, MEDIUM_BRIGHTNESS, HIGH_BRIGHTNESS]:
            n_center = len(segments[center][brightness])
            idx_center = random.sample(range(n_center), min(num_segments_start, n_center))
            for idx in idx_center:
                segments[center][brightness][idx].fillable = True
                segments[center][brightness][idx].is_start_segment = True

            if len(idx_center) < num_segments_start:
                n_edge = len(segments[edge][brightness])
                idx_edge = random.sample(range(n_edge), min(num_segments_start - len(idx_center), n_edge))
                for idx in idx_edge:
                    segments[edge][brightness][idx].fillable = True
                    segments[edge][brightness][idx].is_start_segment = True

        return (
            segments[center][LOW_BRIGHTNESS]
            + segments[edge][LOW_BRIGHTNESS]
            + segments[center][MEDIUM_BRIGHTNESS]
            + segments[edge][MEDIUM_BRIGHTNESS]
            + segments[center][HIGH_BRIGHTNESS]
            + segments[edge][HIGH_BRIGHTNESS]
        )

    @staticmethod
    def _get_image_center(metadata: MosaicMetadata) -> Tuple[int, int, int, int]:
        m_row = int(metadata.n_rows / 4)
        m_col = int(metadata.n_cols / 4)
        return m_row, m_col, m_row * 3, m_col * 3

    def get_mosaic_metadata(self, mosaic_id: str) -> dict:
        metadata = self.persistence.read_mosaic_metadata(mosaic_id)
        json_metadata = json.loads(metadata.json())
        stats = self.persistence.get_segment_stats(mosaic_id)
        json_metadata["dark_segments_left"] = stats[0]
        json_metadata["medium_segments_left"] = stats[1]
        json_metadata["bright_segments_left"] = stats[2]
        self.persistence.disconnect()
        return json_metadata

    def get_mosaic_original_jpeg(self, mosaic_id: str) -> bytes:
        original_jpeg = self.persistence.read_raw_image(mosaic_id, RAW_IMAGE_ORIGINAL_JPEG).image_bytes
        self.persistence.disconnect()
        return original_jpeg

    def get_mosaic_current_jpeg(self, mosaic_id: str) -> bytes:
        current_jpeg = self.persistence.read_raw_image(mosaic_id, RAW_IMAGE_CURRENT_JPEG).image_bytes
        self.persistence.disconnect()
        return current_jpeg

    def get_mosaic_filling_gif(self, mosaic_id: str) -> bytes:
        gif = self.persistence.read_raw_image(mosaic_id, RAW_IMAGE_FILLING_GIF).image_bytes
        self.persistence.disconnect()
        return gif

    def get_mosaic_list(self, filter_by: str) -> List[dict]:
        mosaic_list = self.persistence.read_mosaic_list()
        results = []
        for mosaic_id, index, active, filled, original in mosaic_list:
            if (
                (filter_by == "ACTIVE" and active)
                or (filter_by == "FILLED" and filled)
                or (filter_by == "ORIGINAL" and original)
                or (filter_by == "ALL")
            ):
                results.append({"id": mosaic_id, "index": index})
        self.persistence.disconnect()
        return results

    def delete_mosaic(self, mosaic_id: str):
        self.persistence.delete_mosaic_metadata(mosaic_id)
        self.persistence.commit()
        self.persistence.disconnect()

    def reset_mosaic(self, mosaic_id: str) -> str:
        # load current mosaic data
        metadata = self.persistence.read_mosaic_metadata(mosaic_id)
        orig_pixels = self.persistence.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_ORIGINAL)
        current_pixels = self.persistence.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_CURRENT)
        segments = self.persistence.get_segments(mosaic_id=mosaic_id, filled=1)

        # reset metadata
        metadata.filled = False
        self.persistence.update_mosaic_metadata(metadata)

        # reset current image
        orig_image_pil = np2pil(orig_pixels.pixel_array)
        bg_pil_image = adapt_brightness(orig_image_pil, metadata.mosaic_config.mosaic_background_brightness)
        # type: ignore
        current_pixels.pixel_array = pil2np(bg_pil_image)
        self.persistence.upsert_image_pixels(current_pixels)
        current_jpeg = RawImage(
            mosaic_id=metadata.id, category=RAW_IMAGE_CURRENT_JPEG, image_bytes=pil2bytes(bg_pil_image)
        )
        self.persistence.upsert_raw_image(current_jpeg)

        # reset segments
        for s in segments:
            s.filled = False
            if s.is_start_segment:
                s.fillable = True
            else:
                s.fillable = False
        self.persistence.upsert_segments(segments)
        self.persistence.commit()

        # update filling gif
        gif_image = mosaic_2_gif(persistence=self.persistence, mosaic_id=metadata.id)
        self.persistence.upsert_raw_image(gif_image)
        self.persistence.commit()
        self.persistence.disconnect()

        return mosaic_id
