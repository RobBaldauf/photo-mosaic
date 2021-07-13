import io
import math
import uuid
from typing import Dict, Tuple

import numpy as np
from cv2 import cv2
from fastapi import HTTPException
from PIL import Image, ImageEnhance, ImageFilter
from PIL.ImageStat import Stat

from app.models.app_config import get_config
from app.models.image_pixels import (
    IMAGE_PIXELS_CATEGORY_CURRENT,
    IMAGE_PIXELS_CATEGORY_ORIGINAL,
)
from app.models.raw_image import RAW_IMAGE_FILLING_GIF, RawImage
from app.services.abstract_persistence import AbstractPersistenceService

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


def is_valid_uuid(uuid_string: str) -> bool:
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False


def validate_request_uuid(uuid_string: str, id_label: str) -> str:
    stripped_uuid = str(uuid_string).strip()
    if is_valid_uuid(stripped_uuid):
        return stripped_uuid
    raise HTTPException(
        status_code=400,
        detail=f"Invalid {id_label} id: {stripped_uuid}!",
    )


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


def get_segment_config(width: int, height: int, target_num_segments: int, ratio: Tuple[int, int]) -> Dict[str, int]:
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


def center_crop(image: Image, ratio: Tuple[int, int]) -> Image:
    width, height = image.size
    new_width = int(ratio[0] * height / ratio[1])
    left = (width - new_width) / 2
    top = (height - height) / 2
    right = (width + new_width) / 2
    bottom = (height + height) / 2
    return image.crop((left, top, right, bottom))


def apply_filter(
    portrait_image: Image,
    filter_image: Image,
    blend_value: float,
    blur_radius: float = None,
) -> Image:
    processed_portrait = portrait_image
    processed_filter = filter_image
    if blur_radius:
        processed_filter = processed_filter.filter(ImageFilter.BoxBlur(blur_radius))
    return np2pil(
        cv2.addWeighted(pil2np(processed_portrait), blend_value, pil2np(processed_filter), 1 - blend_value, 0).astype(
            np.uint8
        )
    )


def mosaic_2_gif(
    persistence: AbstractPersistenceService, mosaic_id: str, n_frames_current_image: int = 5, n_frames_filling: int = 5
) -> RawImage:
    GIF_DIMS = (512, 512)
    current_image = persistence.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_CURRENT).pixel_array
    original_image = persistence.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_ORIGINAL).pixel_array

    current_image_small = np2pil(current_image).copy()
    current_image_small.thumbnail(GIF_DIMS)
    current_image_small = current_image_small.quantize(method=GIF_QUANTIZATION_METHOD).convert("P")
    sequence = [current_image_small] * n_frames_current_image

    segments = persistence.get_segments(random_order=True, mosaic_id=mosaic_id, filled=0)
    n = int(len(segments) / n_frames_filling)
    i = 0

    while i < len(segments) + n:
        segs = segments[i : min(i + n, len(segments))]
        orig_mask = np.zeros(original_image.shape, dtype="uint8")
        for s in segs:
            orig_mask[s.y_min : s.y_max, s.x_min : s.x_max, :] = 1
        current_image = np.where(orig_mask == 1, original_image, current_image)
        frame = np2pil(current_image)
        frame.thumbnail(GIF_DIMS)
        sequence.append(frame.quantize(method=GIF_QUANTIZATION_METHOD).convert("P"))
        i += n

    img_byte_arr = io.BytesIO()
    sequence[0].save(img_byte_arr, format="GIF", save_all=True, append_images=sequence[1:], duration=500, loop=0)
    img_byte_arr.seek(0)

    return RawImage(mosaic_id=mosaic_id, category=RAW_IMAGE_FILLING_GIF, image_bytes=img_byte_arr.getvalue())


def generate_id() -> str:
    return str(uuid.uuid4())
