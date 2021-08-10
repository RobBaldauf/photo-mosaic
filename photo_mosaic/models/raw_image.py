from pydantic import BaseModel

RAW_IMAGE_ORIGINAL_JPEG = 0
RAW_IMAGE_CURRENT_JPEG = 1
RAW_IMAGE_FILLING_GIF = 2
RAW_IMAGE_CURRENT_SMALL_JPEG = 3


class RawImage(BaseModel):
    """Model for binary images (e.g. jpeg, gif)"""

    mosaic_id: str
    category: int
    image_bytes: bytes
