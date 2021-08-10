import io
import math
from typing import Dict, Tuple

import numpy as np
from cv2 import cv2
from PIL import Image, ImageEnhance, ImageFilter
from PIL.ImageStat import Stat

from photo_mosaic.models.app_config import get_config
from photo_mosaic.models.mosaic_metadata import MosaicMetadata

INVALID_BRIGHTNESS = -1
LOW_BRIGHTNESS = 0
MEDIUM_BRIGHTNESS = 1
HIGH_BRIGHTNESS = 2
GIF_QUANTIZATION_METHOD = Image.FASTOCTREE


def bytes2pil(byte_arr: bytes) -> Image:
    return Image.open(io.BytesIO(byte_arr))


def pil2bytes(image: Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return img_byte_arr.getvalue()


def np2pil(array: np.ndarray) -> Image:
    return Image.fromarray(array)


def pil2np(image: Image) -> np.ndarray:
    return np.array(image)


def adapt_brightness(image: Image, brightness: float) -> Image:
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness)


def get_brightness_category(image: Image.Image) -> int:
    avg_brightness = get_average_brightness(image)
    if get_config().high_brightness_min < avg_brightness <= get_config().high_brightness_max:
        return HIGH_BRIGHTNESS
    if get_config().medium_brightness_min < avg_brightness <= get_config().medium_brightness_max:
        return MEDIUM_BRIGHTNESS
    if get_config().low_brightness_min <= avg_brightness <= get_config().low_brightness_max:
        return LOW_BRIGHTNESS
    return INVALID_BRIGHTNESS


def get_average_brightness(img: Image.Image) -> float:
    temp_img = img.convert("L")
    stat = Stat(temp_img)
    return stat.mean[0]


def center_crop(image: Image, ratio: Tuple[int, int]) -> Image:
    width, height = image.size
    new_width = int(ratio[0] * height / ratio[1])
    left = (width - new_width) / 2
    top = (height - height) / 2
    right = (width + new_width) / 2
    bottom = (height + height) / 2
    return image.crop((left, top, right, bottom))


def get_image_center(metadata: MosaicMetadata) -> Tuple[int, int, int, int]:
    """
    Get the center area of an image (covering the 2. and 3. quarter of the image in both width and height)
    Args:
        metadata: The mosaic metadata

    Returns: row_min, col_min, row_max, col_max

    """
    m_row = int(metadata.n_rows / 4)
    m_col = int(metadata.n_cols / 4)
    return m_row, m_col, m_row * 3, m_col * 3


def get_segment_config(width: int, height: int, target_num_segments: int, ratio: Tuple[int, int]) -> Dict[str, int]:
    """
    Optimization method to find the best segment config for a mosaic.
    For a given mosaic width, height, target number of segments and segment ratio, determine the
    segment width/height and the number of rows/columns that minimizes the area of unfilled pixels as well as the
    deviation from the given target number of segments.
    Args:
        width: The mosaic width
        height: The mosaic height
        target_num_segments: The desired number of segments in the mosaic
        ratio: The width/heigth ratio of a segment

    Returns: The best found segment config as a dict

    """
    # find smallest ratio using gcd
    gcd = math.gcd(ratio[0], ratio[1])
    smallest_ratio = (ratio[0] / gcd, ratio[1] / gcd)

    # find best config (seh_w, seg_h) that minimizes the score
    # (close to the target_num_segments and minimal unfilled pixel area)
    current_num_seg = int(width * height)
    best_config: Dict[str, int] = {}
    i = 1
    while current_num_seg > target_num_segments / 4:
        # calculate next seg_w, seg_h to try
        seg_w = smallest_ratio[0] * i
        seg_h = smallest_ratio[1] * i

        # calculate rows, cols, unfilled area, score
        unfilled_pixels_w = width % seg_w
        num_cols = (width - unfilled_pixels_w) / seg_w
        unfilled_pixels_h = height % seg_h
        num_rows = (height - unfilled_pixels_h) / seg_h
        unfilled_pixel_area = unfilled_pixels_w * unfilled_pixels_h
        current_num_seg = int(num_cols * num_rows)
        current_config = {
            "seg_h": int(seg_h),
            "seg_w": int(seg_w),
            "num_rows": int(num_rows),
            "num_cols": int(num_cols),
            "num_seg": int(current_num_seg),
            "unfilled_pixel_area": int(unfilled_pixel_area),
            "score": int(
                abs(current_num_seg - target_num_segments)
                + get_config().unused_pixel_area_weight * unfilled_pixel_area / (width * height)
            ),
        }
        # check if score is better than best config
        if len(best_config) > 0:
            if current_config["score"] < best_config["score"]:
                best_config = current_config
        else:
            best_config = current_config
        i += 1
    return best_config


def apply_filter(
    portrait_image: Image,
    filter_image: Image,
    blend_value: float,
    blur_radius: float = None,
) -> Image:
    """
    Merge a given uploaded portrait image with a mosaic segment and create a stylized result image
    Args:
        portrait_image: The uploaded portrait image
        filter_image: The segment to apply as a filter
        blend_value: The blend value for the merge
        blur_radius: The blur radius to apply to the segment before merge

    Returns: The stylized image

    """
    processed_portrait = portrait_image
    processed_filter = filter_image
    if blur_radius:
        processed_filter = processed_filter.filter(ImageFilter.BoxBlur(blur_radius))
    return np2pil(
        cv2.addWeighted(pil2np(processed_portrait), blend_value, pil2np(processed_filter), 1 - blend_value, 0).astype(
            np.uint8
        )
    )
