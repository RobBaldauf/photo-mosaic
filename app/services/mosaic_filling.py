from typing import List, Optional, Tuple

from fastapi import HTTPException

from app.models.app_config import get_config
from app.models.image_pixels import (
    IMAGE_PIXELS_CATEGORY_CURRENT,
    IMAGE_PIXELS_CATEGORY_ORIGINAL,
)
from app.models.raw_image import (
    RAW_IMAGE_CURRENT_JPEG,
    RAW_IMAGE_ORIGINAL_JPEG,
    RawImage,
)
from app.models.segment import Segment
from app.services.abstract_persistence import AbstractPersistenceService
from app.services.mosaic_management import MosaicManagementService
from app.services.nsfw import NSFWService
from app.utils.data import (
    HIGH_BRIGHTNESS,
    LOW_BRIGHTNESS,
    MEDIUM_BRIGHTNESS,
    apply_filter,
    bytes2pil,
    center_crop,
    get_brightness_category,
    mosaic_2_gif,
    np2pil,
    pil2bytes,
)


class MosaicFillingService:
    def __init__(self, persistence_service: AbstractPersistenceService, nsfw_service: NSFWService = None):
        self.persistence = persistence_service
        self.nsfw_service = nsfw_service
        self.mgmt_service = MosaicManagementService(persistence_service)

    async def get_segment_sample(self, mosaic_id: str, query_image_bytes: bytes) -> Tuple[bytes, str]:
        query_image = bytes2pil(query_image_bytes)

        query_image = query_image.convert("RGB")
        width, height = query_image.size

        # find segments by brightness
        brightness = get_brightness_category(query_image)
        segments = self._get_segment_sample(brightness, mosaic_id, 1)

        # apply filter
        seg = segments[0]
        orig_pixels = self.persistence.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_ORIGINAL)
        metadata = self.persistence.read_mosaic_metadata(mosaic_id)
        segment_data = orig_pixels.pixel_array[seg.y_min : seg.y_max, seg.x_min : seg.x_max]
        resized_segment = np2pil(segment_data).resize((width, height))
        if seg.brightness == HIGH_BRIGHTNESS:
            blur_radius = metadata.mosaic_config.segment_blur_high
        elif seg.brightness == MEDIUM_BRIGHTNESS:
            blur_radius = metadata.mosaic_config.segment_blur_medium
        else:
            blur_radius = metadata.mosaic_config.segment_blur_low
        filtered_image = apply_filter(
            portrait_image=query_image,
            filter_image=resized_segment,
            blend_value=metadata.mosaic_config.segment_blend_value,
            blur_radius=blur_radius,
        )

        return pil2bytes(filtered_image), seg.id

    async def fill_random_segment(self, mosaic_id: str, query_image_bytes: bytes, quick_fill: bool):
        num_segments = 5 if quick_fill else 1
        for _ in range(num_segments):
            query_image = bytes2pil(query_image_bytes)
            brightness = get_brightness_category(query_image)
            segments = self._get_segment_sample(brightness, mosaic_id, 1)
            await self.fill_segment(mosaic_id, query_image_bytes, segments[0].id)

    async def fill_segment(self, mosaic_id: str, query_image_bytes: bytes, segment_id: str):
        query_image = bytes2pil(query_image_bytes)
        query_image = query_image.convert("RGB")
        # center crop image
        ratio = (get_config().segment_ratio_width, get_config().segment_ratio_height)
        query_image = center_crop(query_image, ratio)

        # check image for adult content
        if self.nsfw_service:
            if self.nsfw_service.image_is_nsfw(query_image):
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded image contains adult content and can not be added to the mosaic!",
                )

        # get provided segment from db
        segments = self.persistence.get_segments(id=segment_id)
        if len(segments) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"The given segment {segment_id} doesn't exist.",
            )

        seg = segments[0]

        # apply filter
        orig_pixels = self.persistence.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_ORIGINAL)
        current_pixels = self.persistence.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_CURRENT)
        metadata = self.persistence.read_mosaic_metadata(mosaic_id)
        segment_data = orig_pixels.pixel_array[seg.y_min : seg.y_max, seg.x_min : seg.x_max]
        resized_query_image = query_image.resize((metadata.segment_width, metadata.segment_height))
        filtered_image = apply_filter(
            portrait_image=resized_query_image,
            filter_image=segment_data,
            blend_value=metadata.mosaic_config.mosaic_blend_value,
        )

        # update current image pixels
        current_pixels.pixel_array[seg.y_min : seg.y_max, seg.x_min : seg.x_max] = filtered_image
        self.persistence.upsert_image_pixels(current_pixels)

        # update current image jpeg
        current_jpeg = RawImage(
            mosaic_id=metadata.id,
            category=RAW_IMAGE_CURRENT_JPEG,
            image_bytes=pil2bytes(np2pil(current_pixels.pixel_array)),
        )
        self.persistence.upsert_raw_image(current_jpeg)

        if metadata.filled:
            # in case the mosaic has already been filled, but a user is still using the app with this mosaic
            # the segment will just be updated without touching the other segments
            self.persistence.upsert_segments([seg])
            self.persistence.commit()
        else:
            # create list of neighbour segments (to make them fillable)
            neigh_segs = self._get_neighbours(seg, metadata)

            # update segment and neighbours in db
            seg.filled = True
            seg.fillable = False
            for s in neigh_segs:
                if not s.filled:
                    s.fillable = True
            neigh_segs.append(seg)
            self.persistence.upsert_segments(neigh_segs)
            self.persistence.commit()

        # update filling gif
        gif_image = mosaic_2_gif(persistence=self.persistence, mosaic_id=mosaic_id)
        self.persistence.upsert_raw_image(gif_image)
        self.persistence.commit()

        # check whether the total number of available segments is smaller then the required minimum
        # mark the mosaic as inactive and filled
        stats = self.persistence.get_segment_stats(mosaic_id)
        if stats[0] + stats[1] + stats[2] < get_config().num_segments_min:
            await self._handle_mosaic_end(metadata)
        self.persistence.disconnect()

    async def _handle_mosaic_end(self, metadata):
        metadata.filled = True
        metadata.active = False
        self.persistence.update_mosaic_metadata(metadata)
        self.persistence.commit()

        # check if other fillable mosaics are available
        next_active_mosaic_id = self.get_active_mosaic_id()
        if next_active_mosaic_id:
            # set next fillable mosaic to active
            new_active_mosaic = self.persistence.read_mosaic_metadata(next_active_mosaic_id)
            new_active_mosaic.active = True
            self.persistence.update_mosaic_metadata(new_active_mosaic)
            self.persistence.commit()
        else:
            # no fillable mosaics are available -> clone all original mosaics and set the first of them as active
            mosaic_list = self.persistence.read_mosaic_list()
            original_mosaics = []
            for m_id, _, _, _, original in mosaic_list:
                if original:
                    original_mosaics.append(m_id)
            for i, m_id in enumerate(original_mosaics):
                original_bytes = self.persistence.read_raw_image(m_id, RAW_IMAGE_ORIGINAL_JPEG).image_bytes
                new_id = await self.mgmt_service.create_mosaic(original_bytes, metadata.mosaic_config)
                new_metadata = self.persistence.read_mosaic_metadata(new_id)
                new_metadata.original = False
                if i == 0:
                    new_metadata.active = True
                self.persistence.update_mosaic_metadata(new_metadata)
                self.persistence.commit()

    def get_active_mosaic_id(self) -> Optional[str]:
        """
        Returns the active mosaic. If no active mosaic is found, return the next fillable mosaic.
        If no fillable mosaic is found return None.
        """
        mosaic_list = self.persistence.read_mosaic_list()
        new_active = None
        for mosaic_id, _, active, filled, _ in mosaic_list:
            if active:
                return mosaic_id
            if not filled and not new_active:
                new_active = mosaic_id
        return new_active

    def _get_segment_sample(self, brightness: int, mosaic_id: str, sample_size: int) -> List[Segment]:
        segments = self.persistence.get_segments(
            limit=sample_size,
            mosaic_id=mosaic_id,
            brightness=brightness,
            random_order=True,
            fillable=True,
        )
        if len(segments) >= sample_size:
            # enough segments for detected brightness
            return segments

        # not enough segments for detected brightness (use segments from other brightnesses)
        next_brightness = MEDIUM_BRIGHTNESS
        if brightness == MEDIUM_BRIGHTNESS:
            next_brightness = LOW_BRIGHTNESS
        segments.extend(
            self.persistence.get_segments(
                limit=sample_size - len(segments),
                mosaic_id=mosaic_id,
                brightness=next_brightness,
                random_order=True,
                fillable=True,
            )
        )
        if len(segments) >= sample_size:
            return segments

        # still not enough segments for detected brightness (use segments from last brightness)
        next_brightness = HIGH_BRIGHTNESS
        if brightness == HIGH_BRIGHTNESS:
            next_brightness = LOW_BRIGHTNESS
        segments.extend(
            self.persistence.get_segments(
                limit=sample_size - len(segments),
                mosaic_id=mosaic_id,
                brightness=next_brightness,
                random_order=True,
                fillable=True,
            )
        )
        return segments

    def _get_neighbours(self, segment, metadata) -> List[Segment]:
        segs = []
        for r in [-1, 0, 1]:
            n_r = segment.row_idx + r
            for c in [-1, 0, 1]:
                n_c = segment.col_idx + c
                if 0 <= n_r < metadata.n_rows and 0 <= n_c < metadata.n_cols:
                    segs.append(self.persistence.get_segments(mosaic_id=metadata.id, row_idx=n_r, col_idx=n_c)[0])
        return segs
