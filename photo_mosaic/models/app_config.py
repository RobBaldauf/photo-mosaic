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

    # db config
    sql_lite_path: str

    # Mosaic config
    original_image_max_size: int
    gif_image_max_size: int
    sample_image_max_size: int
    unused_pixel_area_weight: int

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

    # mail config
    mail_username: str
    mail_password: str
    mail_from: str
    mail_port: int
    mail_server: str
    mail_tls: bool
    mail_ssl: bool
    mail_use_credentials: bool
    mail_validate_certs: bool

    class Config:
        env_file = "config/.env"


@lru_cache()
def get_config():
    return AppConfig()
