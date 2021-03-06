from functools import lru_cache
from typing import List

from pydantic import BaseSettings


class AppConfig(BaseSettings):
    """Model holding the app configuration"""

    # api config
    enable_documentation: bool
    cors_origins: List[str]

    # auth config
    enable_auth: bool
    jwt_secret: str

    # nsfw config
    enable_nsfw_content_filter: bool
    nsfw_model_path: str

    # persistence config
    sqlite_path: str
    uploaded_image_path: str

    # Mosaic config
    original_image_max_size: int
    gif_image_max_size: int
    sample_image_max_size: int
    current_image_thumbnail_size: int
    unused_pixel_area_weight: int
    segment_sample_size: int

    # segment config
    num_segments_start: int
    num_segments_min: int
    segment_ratio_width: int
    segment_ratio_height: int
    low_brightness_min: int
    low_brightness_max: int
    medium_brightness_min: int
    medium_brightness_max: int
    high_brightness_min: int
    high_brightness_max: int

    class Config:
        env_file = "config/.env"


@lru_cache()
def get_config():
    return AppConfig()
