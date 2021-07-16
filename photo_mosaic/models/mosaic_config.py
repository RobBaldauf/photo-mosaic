from pydantic import BaseModel


class MosaicConfig(BaseModel):
    """Configuration that specifies the style of a mosaic, should be provided by the admin on mosaic creation"""

    num_segments: int
    mosaic_background_brightness: float
    mosaic_blend_value: float
    segment_blend_value: float
    segment_blur_low: float
    segment_blur_medium: float
    segment_blur_high: float
