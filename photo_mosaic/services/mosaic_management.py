import json
import random
import uuid
from typing import List, Optional, Tuple

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
    RAW_IMAGE_CURRENT_SMALL_JPEG,
    RAW_IMAGE_FILLING_GIF,
    RAW_IMAGE_ORIGINAL_JPEG,
    RawImage,
)
from photo_mosaic.models.segment import Segment
from photo_mosaic.services.persistence import db
from photo_mosaic.utils.animation import mosaic_2_gif
from photo_mosaic.utils.image_processing import (
    HIGH_BRIGHTNESS,
    LOW_BRIGHTNESS,
    MEDIUM_BRIGHTNESS,
    adapt_brightness,
    bytes2pil,
    get_brightness_category,
    get_image_center,
    get_segment_config,
    np2pil,
    pil2bytes,
    pil2np,
)
from photo_mosaic.utils.request_validation import generate_id


class MosaicManagementService:
    """Service for creation, modification, deletion and retrieval of mosaic data"""

    def create_mosaic(self, image_bytes: bytes, config: MosaicConfig) -> str:
        """
        Create a new mosaic image (including its metadata, np pixel_arrays, segements and binary image files)
        Args:
            image_bytes: The binary image to create a mosaic from
            config: The configuration for the mosaic creation process

        Returns: The mosaic id (UUID)

        """
        # create metadata + original image
        metadata, image = self._create_mosaic_metadata(image_bytes, config)
        db.insert_mosaic_metadata(metadata)

        # create original jpeg
        original = image.copy()
        original.thumbnail((get_config().original_image_max_size, get_config().original_image_max_size))
        original_jpeg = RawImage(
            mosaic_id=metadata.id, category=RAW_IMAGE_ORIGINAL_JPEG, image_bytes=pil2bytes(original)
        )
        db.upsert_raw_image(original_jpeg)

        # create original image pixels (np)
        original_pixels = ImagePixels(
            mosaic_id=metadata.id, category=IMAGE_PIXELS_CATEGORY_ORIGINAL, pixel_array=pil2np(image)
        )
        db.upsert_image_pixels(original_pixels)

        # create current jpeg
        bg_pil_image = adapt_brightness(image, config.mosaic_bg_brightness)
        current_jpeg = RawImage(
            mosaic_id=metadata.id, category=RAW_IMAGE_CURRENT_JPEG, image_bytes=pil2bytes(bg_pil_image)
        )
        db.upsert_raw_image(current_jpeg)

        # create current image pixels (np)
        current_pixels = ImagePixels(
            mosaic_id=metadata.id, category=IMAGE_PIXELS_CATEGORY_CURRENT, pixel_array=pil2np(bg_pil_image)
        )
        db.upsert_image_pixels(current_pixels)

        # create current jpeg thumbnail
        bg_pil_image.thumbnail((get_config().current_image_thumbnail_size, get_config().current_image_thumbnail_size))
        current_jpeg_small = RawImage(
            mosaic_id=metadata.id, category=RAW_IMAGE_CURRENT_SMALL_JPEG, image_bytes=pil2bytes(bg_pil_image)
        )
        db.upsert_raw_image(current_jpeg_small)

        # create segments
        segments = self._create_segments(metadata, original_pixels)
        db.upsert_segments(segments)

        # Commit changes to db
        db.commit()

        # create filling gif animation
        gif_image = mosaic_2_gif(mosaic_id=metadata.id)
        db.upsert_raw_image(gif_image)
        db.commit()
        return metadata.id

    @staticmethod
    def _create_mosaic_metadata(image_bytes: bytes, mosaic_cfg: MosaicConfig) -> Tuple[MosaicMetadata, Image]:
        """
        Create the metadata object for mosaic
        Args:
            image_bytes: The binary image to create a mosaic from
            mosaic_cfg: The configuration for the mosaic creation process

        Returns: (the metadata object, the query image)

        """
        # if no other mosaic exists set this to active
        is_active = db.mosaic_count() == 0

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

    @staticmethod
    def _create_segments(metadata: MosaicMetadata, pixels: ImagePixels) -> List[Segment]:
        """
        Create the segments (and their metadata) for a mosaic image based on the mosaic metadata.
        Segments will be separated based on their brightness (low,medium,high).
        Random segments will be chosen as start segments for the filling process.
        Args:
            metadata: The mosaic metadata object
            pixels: The original image pixels

        Returns: A list of segment objects

        """
        # calculate the center of the image
        row_min, col_min, row_max, col_max = get_image_center(metadata)
        center = "center"
        edge = "edge"
        segments: dict = {center: {0: [], 1: [], 2: []}, edge: {0: [], 1: [], 2: []}}

        random_sort_keys = random.sample(range(metadata.n_cols * metadata.n_rows), metadata.n_cols * metadata.n_rows)
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
                    random_sort_key=random_sort_keys.pop(0),
                )
                np_segment = pixels.pixel_array[new_seg.y_min : new_seg.y_max, new_seg.x_min : new_seg.x_max]
                new_seg.brightness = get_brightness_category(np2pil(np_segment))

                if row_min <= r <= row_max and col_min <= c <= col_max:
                    position = center
                else:
                    position = edge
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
    def get_mosaic_metadata(mosaic_id: str) -> dict:
        metadata = db.read_mosaic_metadata(mosaic_id)
        json_metadata = json.loads(metadata.json())
        stats = db.get_segment_stats(mosaic_id)
        json_metadata["dark_segments_left"] = stats[0]
        json_metadata["medium_segments_left"] = stats[1]
        json_metadata["bright_segments_left"] = stats[2]
        return json_metadata

    @staticmethod
    def get_mosaic_original_jpeg(mosaic_id: str) -> bytes:
        return db.read_raw_image(mosaic_id, RAW_IMAGE_ORIGINAL_JPEG).image_bytes

    @staticmethod
    def get_mosaic_current_jpeg(mosaic_id: str) -> bytes:
        return db.read_raw_image(mosaic_id, RAW_IMAGE_CURRENT_JPEG).image_bytes

    @staticmethod
    def get_mosaic_current_jpeg_thumbnail(mosaic_id: str) -> bytes:
        return db.read_raw_image(mosaic_id, RAW_IMAGE_CURRENT_SMALL_JPEG).image_bytes

    @staticmethod
    def get_mosaic_filling_gif(mosaic_id: str) -> bytes:
        return db.read_raw_image(mosaic_id, RAW_IMAGE_FILLING_GIF).image_bytes

    @staticmethod
    def get_mosaic_list(filter_by: str) -> List[dict]:
        """
        Get a list of mosaics
        Args:
            filter_by: The filter criterion ("ACTIVE": active mosaics,
                                             "FILLED": fully filled mosaics,
                                             "ORIGINAL": mosaics created by the admin (not automatically by the API),
                                             "ALL": all mosaics)

        Returns: A list of dictionaries each containing id, index and title for a mosaic

        """
        mosaic_list = db.read_mosaic_list()
        results = []
        for mosaic_id, index, title, active, filled, original in mosaic_list:
            if (
                (filter_by == "ACTIVE" and active)
                or (filter_by == "FILLED" and filled)
                or (filter_by == "ORIGINAL" and original)
                or (filter_by == "ALL")
            ):
                results.append({"id": mosaic_id, "index": index, "title": title})
        return results

    def delete_mosaic(self, mosaic_id: str):
        """
        Delete a mosaic from the db and set next mosaic active
        Args:
            mosaic_id: The mosaic id

        Raises:
            HTTPException: Mosaic does not exist

        """
        if not db.mosaic_exists(mosaic_id):
            raise HTTPException(status_code=404, detail=f"Mosaic {mosaic_id} does not exist.")
        metadata = db.read_mosaic_metadata(mosaic_id)
        db.delete_mosaic_metadata(mosaic_id)
        db.commit()
        if metadata.active:
            self.set_next_mosaic_active(metadata)

    @staticmethod
    def reset_mosaic(mosaic_id: str) -> str:
        """
        Reset a mosaic to its state right after creation.
        All associated RawImage, ImagePixels and Segment data structures will be cleared.
        Args:
            mosaic_id: The mosaic id

        Returns: The mosaic id

        """
        # load current mosaic data
        metadata = db.read_mosaic_metadata(mosaic_id)
        orig_pixels = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_ORIGINAL)
        current_pixels = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_CURRENT)
        segments = db.get_segments(mosaic_id=mosaic_id)

        # reset metadata
        metadata.filled = False
        db.update_mosaic_metadata(metadata)

        # reset current image
        orig_image_pil = np2pil(orig_pixels.pixel_array)
        bg_pil_image = adapt_brightness(orig_image_pil, metadata.mosaic_config.mosaic_bg_brightness)
        # type: ignore
        current_pixels.pixel_array = pil2np(bg_pil_image)
        db.upsert_image_pixels(current_pixels)
        current_jpeg = RawImage(
            mosaic_id=metadata.id, category=RAW_IMAGE_CURRENT_JPEG, image_bytes=pil2bytes(bg_pil_image)
        )
        db.upsert_raw_image(current_jpeg)

        # create current thumbnail
        bg_pil_image.thumbnail((get_config().current_image_thumbnail_size, get_config().current_image_thumbnail_size))
        current_jpeg_small = RawImage(
            mosaic_id=metadata.id, category=RAW_IMAGE_CURRENT_SMALL_JPEG, image_bytes=pil2bytes(bg_pil_image)
        )
        db.upsert_raw_image(current_jpeg_small)

        # reset segments
        for s in segments:
            s.filled = False
            if s.is_start_segment:
                s.fillable = True
            else:
                s.fillable = False
        db.upsert_segments(segments)
        db.commit()

        # update filling gif
        gif_image = mosaic_2_gif(mosaic_id=metadata.id)
        db.upsert_raw_image(gif_image)
        db.commit()

        return mosaic_id

    @staticmethod
    def finish_mosaic(metadata: MosaicMetadata):
        """
        Set inactive, fill remaining segments and create final jpegs

        Args:
            metadata: The metadata of the mosaic that shall be terminated

        """
        metadata.filled = True
        metadata.active = False
        db.update_mosaic_metadata(metadata)
        db.commit()

        # fill remaining unfilled segments with original data and update artefacts
        orig_pixels = db.read_image_pixels(metadata.id, IMAGE_PIXELS_CATEGORY_ORIGINAL)
        current_pixels = db.read_image_pixels(metadata.id, IMAGE_PIXELS_CATEGORY_CURRENT)
        segments = db.get_segments(mosaic_id=metadata.id, filled=False, fillable=True)
        for seg in segments:
            segment_data = orig_pixels.pixel_array[seg.y_min : seg.y_max, seg.x_min : seg.x_max]
            current_pixels.pixel_array[seg.y_min : seg.y_max, seg.x_min : seg.x_max] = segment_data
        db.upsert_image_pixels(current_pixels)
        cur_pil = np2pil(current_pixels.pixel_array)
        current_jpeg = RawImage(
            mosaic_id=metadata.id,
            category=RAW_IMAGE_CURRENT_JPEG,
            image_bytes=pil2bytes(cur_pil),
        )
        db.upsert_raw_image(current_jpeg)

        # update current image jpeg thumbnail
        cur_pil.thumbnail((get_config().current_image_thumbnail_size, get_config().current_image_thumbnail_size))
        current_jpeg_small = RawImage(
            mosaic_id=metadata.id, category=RAW_IMAGE_CURRENT_SMALL_JPEG, image_bytes=pil2bytes(cur_pil)
        )
        db.upsert_raw_image(current_jpeg_small)

    def set_next_mosaic_active(self, metadata: MosaicMetadata):
        """
        Try to find the next mosaic, if None is available clone the existing original mosaics and set
        the first of them as active

        Args:
            metadata: The metadata of the mosaic that shall be terminated

        """
        # check if other fillable mosaics are available
        next_active_mosaic_id = self.get_next_mosaic_id()
        if next_active_mosaic_id:
            # set next fillable mosaic to active
            new_active_mosaic = db.read_mosaic_metadata(next_active_mosaic_id)
            new_active_mosaic.active = True
            db.update_mosaic_metadata(new_active_mosaic)
            db.commit()
        else:
            # no fillable mosaics are available -> clone all original mosaics and set the first of them as active
            # to ensure endless filling of mosaics
            mosaic_list = db.read_mosaic_list()
            original_mosaics = []
            for m_id, _, _, _, _, original in mosaic_list:
                if original:
                    original_mosaics.append(m_id)
            for i, m_id in enumerate(original_mosaics):
                original = db.read_image_pixels(m_id, IMAGE_PIXELS_CATEGORY_ORIGINAL)
                original = pil2bytes(np2pil(original.pixel_array))
                new_id = mgmt_service.create_mosaic(original, metadata.mosaic_config)

                new_metadata = db.read_mosaic_metadata(new_id)
                new_metadata.original = False
                if i == 0:
                    new_metadata.active = True
                db.update_mosaic_metadata(new_metadata)
                db.commit()

    @staticmethod
    def get_next_mosaic_id() -> Optional[str]:
        """
        Returns the active mosaic. If no active mosaic is found, return the next fillable mosaic.
        If no fillable mosaic is found return None.
        """
        mosaic_list = db.read_mosaic_list()
        for mosaic_id, _, _, active, filled, _ in mosaic_list:
            if not filled and not active:
                return mosaic_id
        return None


mgmt_service = MosaicManagementService()
