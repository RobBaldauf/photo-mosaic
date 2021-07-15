from pydantic import BaseModel

from photo_mosaic.models.mosaic_config import MosaicConfig


class MosaicMetadata(BaseModel):
    id: str
    idx: int
    active: bool
    filled: bool
    original: bool
    segment_width: int
    segment_height: int
    n_rows: int
    n_cols: int
    space_top: int
    space_left: int
    mosaic_config: MosaicConfig
