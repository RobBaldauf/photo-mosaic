from typing import List, Optional, Tuple

from fastapi import HTTPException
from PIL.Image import Image

from photo_mosaic.models.app_config import get_config
from photo_mosaic.models.image_pixels import (
    IMAGE_PIXELS_CATEGORY_CURRENT,
    IMAGE_PIXELS_CATEGORY_ORIGINAL,
)
from photo_mosaic.models.mosaic_metadata import MosaicMetadata
from photo_mosaic.models.raw_image import (
    RAW_IMAGE_CURRENT_JPEG,
    RAW_IMAGE_ORIGINAL_JPEG,
    RawImage,
)
from photo_mosaic.models.segment import Segment
from photo_mosaic.services.mosaic_management import mgmt_service
from photo_mosaic.services.nsfw import NSFWService
from photo_mosaic.services.persistence import db
from photo_mosaic.utils.image_processing import (
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
    """Service for sampling/filling mosaic segments"""

    def __init__(self):
        self.nsfw_service = None
        if get_config().enable_nsfw_content_filter:
            self.nsfw_service = NSFWService(get_config().nsfw_model_path)

    def get_segment_sample(self, mosaic_id: str, query_image_bytes: bytes) -> Tuple[bytes, str]:
        """
        For the given query image, find a segment in the given mosaic that has not been filled and matches the query
        image in brightness. Merge this segment with the query image to create stylized version of the query image.
        Args:
            mosaic_id: The mosaic id
            query_image_bytes: The uploaded portrait image to apply a segment filter to

        Returns: (The stylized query image, the id of the used segment)

        """
        query_image = bytes2pil(query_image_bytes)
        query_image = query_image.convert("RGB")
        # Limit sample size to speed up processing
        query_image.thumbnail((get_config().sample_image_max_size, get_config().sample_image_max_size))
        width, height = query_image.size

        # find segments by brightness
        brightness = get_brightness_category(query_image)
        segments = self._get_segment_sample(brightness, mosaic_id, 1)

        # apply filter
        seg = segments[0]
        orig_pixels = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_ORIGINAL)
        metadata = db.read_mosaic_metadata(mosaic_id)
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

    def fill_random_segments(self, mosaic_id: str, query_image_bytes: bytes, quick_fill: bool):
        """
        Fill n random segments from the given mosaic with the provided query image merged with each segment
        Args:
            mosaic_id: The mosaic id
            query_image_bytes: The uploaded portrait image that shall be added to the mosaic
            quick_fill: Determines the number of filled segments (False:n=1, True:n=5)

        """
        num_segments = 5 if quick_fill else 1
        query_image = bytes2pil(query_image_bytes)
        brightness = get_brightness_category(query_image)
        segments = self._get_segment_sample(brightness, mosaic_id, num_segments)
        self._fill_segments(mosaic_id, query_image, segments)

    def fill_segment(self, mosaic_id: str, query_image_bytes: bytes, segment_id: str):
        """
        Fill the provided segment from the given mosaic with the provided query image merged with the segment
        Args:
            mosaic_id: The mosaic id
            query_image_bytes: The uploaded portrait image that shall be added to the mosaic
            segment_id: The id of the segment that shall be used for filling

        """
        query_image = bytes2pil(query_image_bytes)
        segments = db.get_segments(id=segment_id)
        self._fill_segments(mosaic_id, query_image, segments)

    def _fill_segments(self, mosaic_id: str, query_image: Image, segments: List[Segment]):
        # convert + center crop image
        q_image = query_image.copy()
        q_image = q_image.convert("RGB")
        ratio = (get_config().segment_ratio_width, get_config().segment_ratio_height)
        q_image = center_crop(q_image, ratio)

        # check image for adult content
        if self.nsfw_service:
            if self.nsfw_service.image_is_nsfw(q_image):
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded image contains adult content and can not be added to the mosaic!",
                )

        # read image artifacts
        orig_pixels = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_ORIGINAL)
        current_pixels = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_CURRENT)
        metadata = db.read_mosaic_metadata(mosaic_id)

        for seg in segments:
            # apply filter
            segment_data = orig_pixels.pixel_array[seg.y_min : seg.y_max, seg.x_min : seg.x_max]
            resized_query_image = q_image.resize((metadata.segment_width, metadata.segment_height))
            filtered_image = apply_filter(
                portrait_image=resized_query_image,
                filter_image=segment_data,
                blend_value=metadata.mosaic_config.mosaic_blend_value,
            )

            # update current image pixels
            current_pixels.pixel_array[seg.y_min : seg.y_max, seg.x_min : seg.x_max] = filtered_image

            # check if mosaic has already been filled
            # in case the mosaic has already been filled, but a user is still using the photo_mosaic with this mosaic
            # the segment will just be updated without touching the other segments
            if not metadata.filled:
                # create list of neighbour segments (to make them fillable)
                neigh_segs = self._get_neighbours(seg, metadata)

                # update segment and neighbours in db
                seg.filled = True
                seg.fillable = False
                for s in neigh_segs:
                    if not s.filled:
                        s.fillable = True
                neigh_segs.append(seg)
                db.upsert_segments(neigh_segs)
                db.commit()

        # write current pixels back to db
        db.upsert_image_pixels(current_pixels)

        # update current image jpeg
        current_jpeg = RawImage(
            mosaic_id=metadata.id,
            category=RAW_IMAGE_CURRENT_JPEG,
            image_bytes=pil2bytes(np2pil(current_pixels.pixel_array)),
        )
        db.upsert_raw_image(current_jpeg)

        # update filling gif
        gif_image = mosaic_2_gif(mosaic_id=mosaic_id)
        db.upsert_raw_image(gif_image)
        db.commit()

        # check whether the total number of available segments is smaller then the required minimum
        # mark the mosaic as inactive and filled
        stats = db.get_segment_stats(mosaic_id)
        if stats[0] + stats[1] + stats[2] < get_config().num_segments_min:
            self._handle_mosaic_end(metadata)

    def _handle_mosaic_end(self, metadata: MosaicMetadata):
        """
        If the number of segments in a mosaic is lower then the min. number of segments, the mosaic will be terminated.

        Args:
            metadata: The metadata of the mosaic that shall be terminated

        """
        metadata.filled = True
        metadata.active = False
        db.update_mosaic_metadata(metadata)
        db.commit()

        # check if other fillable mosaics are available
        next_active_mosaic_id = self._get_active_mosaic_id()
        if next_active_mosaic_id:
            # set next fillable mosaic to active
            new_active_mosaic = db.read_mosaic_metadata(next_active_mosaic_id)
            new_active_mosaic.active = True
            db.update_mosaic_metadata(new_active_mosaic)
            db.commit()
        else:
            # no fillable mosaics are available -> clone all original mosaics and set the first of them as active
            mosaic_list = db.read_mosaic_list()
            original_mosaics = []
            for m_id, _, _, _, original in mosaic_list:
                if original:
                    original_mosaics.append(m_id)
            for i, m_id in enumerate(original_mosaics):
                original_bytes = db.read_raw_image(m_id, RAW_IMAGE_ORIGINAL_JPEG).image_bytes
                new_id = mgmt_service.create_mosaic(original_bytes, metadata.mosaic_config)
                new_metadata = db.read_mosaic_metadata(new_id)
                new_metadata.original = False
                if i == 0:
                    new_metadata.active = True
                db.update_mosaic_metadata(new_metadata)
                db.commit()

    @staticmethod
    def _get_active_mosaic_id() -> Optional[str]:
        """
        Returns the active mosaic. If no active mosaic is found, return the next fillable mosaic.
        If no fillable mosaic is found return None.
        """
        mosaic_list = db.read_mosaic_list()
        new_active = None
        for mosaic_id, _, active, filled, _ in mosaic_list:
            if active:
                return mosaic_id
            if not filled and not new_active:
                new_active = mosaic_id
        return new_active

    @staticmethod
    def _get_segment_sample(brightness: int, mosaic_id: str, sample_size: int) -> List[Segment]:
        """
        Get a sample of segments with the given brightness from the given mosaic. If not enough segments from the
        required brightness are available, segments from other brightness categories will be used.
        Args:
            brightness: The required brightness (LOW_BRIGHTNESS, MEDIUM_BRIGHTNESS, HIGH_BRIGHTNESS)
            mosaic_id: The mosaic id
            sample_size: The number of segments that shall be returned

        Returns: A list of matching segments

        """
        segments = db.get_segments(
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
            db.get_segments(
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
            db.get_segments(
                limit=sample_size - len(segments),
                mosaic_id=mosaic_id,
                brightness=next_brightness,
                random_order=True,
                fillable=True,
            )
        )
        return segments

    @staticmethod
    def _get_neighbours(segment: Segment, metadata: MosaicMetadata) -> List[Segment]:
        """
        Get the segments that are direct neighbours to a given segment
        Args:
            segment: The segment
            metadata: The mosaic metadata

        Returns: A list of segments

        """
        segs = []
        for r in [-1, 0, 1]:
            n_r = segment.row_idx + r
            for c in [-1, 0, 1]:
                n_c = segment.col_idx + c
                if 0 <= n_r < metadata.n_rows and 0 <= n_c < metadata.n_cols:
                    segs.append(db.get_segments(mosaic_id=metadata.id, row_idx=n_r, col_idx=n_c)[0])
        return segs


filling_service = MosaicFillingService()
