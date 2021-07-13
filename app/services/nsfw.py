import logging
from os.path import exists

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

from app.utils.data import pil2np

IMAGE_DIMS = (224, 224)
DRAWINGS = 0
HENTAI = 1
NEUTRAL = 2
PORN = 3
SEXY = 4
FILTER_THRESHOLD = 0.5


class NSFWService:
    """
    https://github.com/GantMan/nsfw_model
    """

    def __init__(self, model_path: str):
        tf.get_logger().setLevel("ERROR")
        tf.autograph.set_verbosity(3)
        print("Loading NSFW tensorflow model...")
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path):
        if model_path is None or not exists(model_path):
            raise ValueError("saved_model_path must be a valid directory of a saved model to load.")
        model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
        return model

    def image_is_nsfw(self, image: Image):
        img = image.resize(IMAGE_DIMS, Image.NEAREST)
        nd_image = pil2np(img).astype(float)
        nd_image /= 255.0
        nd_image = tf.expand_dims(nd_image, axis=0)
        preds = self.model.predict(nd_image)[0]

        if preds[HENTAI] + preds[PORN] + preds[SEXY] > FILTER_THRESHOLD:
            logging.warning(
                "Attempted NSWF image upload {hentai:%s, porn:%s, sexy:%s}", preds[HENTAI], preds[PORN], preds[SEXY]
            )
            return True
        return False
