from abc import ABC
from typing import Dict, List, Tuple

from app.models.image_pixels import ImagePixels
from app.models.mosaic_metadata import MosaicMetadata
from app.models.raw_image import RawImage
from app.models.segment import Segment


class AbstractPersistenceService(ABC):
    def commit(self):
        raise NotImplementedError()

    def disconnect(self):
        raise NotImplementedError()

    def mosaic_count(self) -> int:
        raise NotImplementedError()

    def read_mosaic_list(self) -> List[Tuple[str, int, bool, bool, bool]]:
        raise NotImplementedError()

    def insert_mosaic_metadata(self, metadata: MosaicMetadata):
        raise NotImplementedError()

    def update_mosaic_metadata(self, metadata: MosaicMetadata):
        raise NotImplementedError()

    def read_mosaic_metadata(self, mosaic_id: str, active_only: bool = False) -> MosaicMetadata:
        raise NotImplementedError()

    def delete_mosaic_metadata(self, mosaic_id: str):
        raise NotImplementedError()

    def upsert_raw_image(self, raw_image: RawImage):
        raise NotImplementedError()

    def read_raw_image(self, mosaic_id: str, category: int) -> RawImage:
        raise NotImplementedError()

    def upsert_image_pixels(self, image_pixels: ImagePixels):
        raise NotImplementedError()

    def read_image_pixels(self, mosaic_id: str, category: int) -> ImagePixels:
        raise NotImplementedError()

    def get_segment_stats(self, mosaic_id: str) -> Dict[int, int]:
        raise NotImplementedError()

    def get_segments(self, limit: int = -1, offset: int = 0, random_order: bool = False, **kwargs) -> List[Segment]:
        raise NotImplementedError()

    def upsert_segments(self, segments: List[Segment]):
        raise NotImplementedError()
