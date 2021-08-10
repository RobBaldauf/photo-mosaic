import io

import numpy as np

from photo_mosaic.models.app_config import get_config
from photo_mosaic.models.image_pixels import (
    IMAGE_PIXELS_CATEGORY_CURRENT,
    IMAGE_PIXELS_CATEGORY_ORIGINAL,
)
from photo_mosaic.models.raw_image import RAW_IMAGE_FILLING_GIF, RawImage
from photo_mosaic.services.persistence import db
from photo_mosaic.utils.image_processing import GIF_QUANTIZATION_METHOD, np2pil


def mosaic_2_gif(mosaic_id: str, n_frames_current_image: int = 5, n_frames_filling: int = 5) -> RawImage:
    """
    For the given mosaic, render a random filling process from the current state to the completely filled image as a gif
    Args:
        mosaic_id: The mosaic id
        n_frames_current_image: The number of gif frames to show the current state
        n_frames_filling: The number of gif frames to show the filling process

    Returns: A gif image

    """
    # Load raw images
    gif_dims = (get_config().gif_image_max_size, get_config().gif_image_max_size)
    current_image = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_CURRENT).pixel_array
    original_image = db.read_image_pixels(mosaic_id, IMAGE_PIXELS_CATEGORY_ORIGINAL).pixel_array

    # reduce raw image dims (to reduce gif size) and create frames for current state
    current_image_small = np2pil(current_image).copy()
    current_image_small.thumbnail(gif_dims)
    current_image_small = current_image_small.quantize(method=GIF_QUANTIZATION_METHOD).convert("P")
    sequence = [current_image_small] * n_frames_current_image

    # Get segments of the mosaic that have not been filled yet
    segments = db.get_segments(random_order=True, mosaic_id=mosaic_id, filled=0)
    n = int(len(segments) / n_frames_filling)
    i = 0

    # for each filling frame
    while i < len(segments) + n:
        # select a subset of segments
        segs = segments[i : min(i + n, len(segments))]

        # add them to the current state of the mosaic (based on the original image)
        orig_mask = np.zeros(original_image.shape, dtype="uint8")
        for s in segs:
            orig_mask[s.y_min : s.y_max, s.x_min : s.x_max, :] = 1
        current_image = np.where(orig_mask == 1, original_image, current_image)

        # reduce the image dims and the frame
        frame = np2pil(current_image)
        frame.thumbnail(gif_dims)
        sequence.append(frame.quantize(method=GIF_QUANTIZATION_METHOD).convert("P"))
        i += n

    # render the gif from the sequence of frames
    img_byte_arr = io.BytesIO()
    sequence[0].save(img_byte_arr, format="GIF", save_all=True, append_images=sequence[1:], duration=500, loop=0)
    img_byte_arr.seek(0)

    return RawImage(mosaic_id=mosaic_id, category=RAW_IMAGE_FILLING_GIF, image_bytes=img_byte_arr.getvalue())
