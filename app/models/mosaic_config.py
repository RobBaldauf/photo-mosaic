from pydantic import BaseModel


class MosaicConfig(BaseModel):
    num_segments: int
    mosaic_background_brightness: float
    mosaic_blend_value: float
    segment_blend_value: float
    segment_blur_low: float
    segment_blur_medium: float
    segment_blur_high: float
