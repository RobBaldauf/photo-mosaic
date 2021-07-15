from typing import Any

import numpy as np
from pydantic import BaseModel

IMAGE_PIXELS_CATEGORY_ORIGINAL = 0
IMAGE_PIXELS_CATEGORY_CURRENT = 1


class NPArray(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> str:
        return v


class ImagePixels(BaseModel):
    mosaic_id: str
    category: int
    pixel_array: NPArray
