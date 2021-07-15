from pydantic import BaseModel

RAW_IMAGE_ORIGINAL_JPEG = 0
RAW_IMAGE_CURRENT_JPEG = 1
RAW_IMAGE_FILLING_GIF = 2
RAW_IMAGE_FILLED_JPEG = 3
RAW_IMAGE_FILLED_SMALL_JPEG = 4


class RawImage(BaseModel):
    mosaic_id: str
    category: int
    image_bytes: bytes
