import functools
from typing import Optional, List, Callable

import cv2
import numpy as np
from PIL import Image

import constants


class CannyImageProcessor:
    def __init__(self, image: Image) -> None:
        self.image = image

    def __pipe(self, data, funcs: List[Callable]):
        return functools.reduce(lambda seed, fun: fun(seed), [data, *funcs])

    def __to_single_channel(self, image) -> Image:
        return image[:, :, None]

    def __to_canny(self, image: Image, lower_threshold: Optional[int], upper_threshold: Optional[int]):
        if lower_threshold is None:
            lower_threshold = constants.CANNY_DEFAULT_LOWER_THRESHOLD
        if upper_threshold is None:
            upper_threshold = constants.CANNY_DEFAULT_UPPER_THRESHOLD
        return cv2.Canny(image, lower_threshold, upper_threshold)

    def apply(self, lower_threshold: Optional[int] = None, upper_threshold: Optional[int] = None) -> Image:
        return self.__pipe(
            self.image,
            [
                np.array,
                lambda img: self.__to_canny(img, lower_threshold, upper_threshold),
                self.__to_single_channel,
                lambda img: np.concatenate([img, img, img], axis=2),
                Image.fromarray
            ]
        )
