import json
import uuid
from pathlib import Path
import random
from typing import List, Dict, Tuple
import PIL
from PIL import PngImagePlugin
from PIL.Image import Image

import constants
from image_size import ImageSize
from sdxl_model import SdxlModel


def load_images(path: str) -> Tuple[List[Image], List[Image]]:
    images = []
    for image in Path(path).glob("*.png"):
        images.append(PIL.Image.open(image))
    return images, images


def save_img(
        imgs: List[Image],
        index: int,
        prompt: str,
        negative: str,
        num_inference_steps: int,
        prompt_fidelity: float,
        seed: int,
        folder_images_path: str):
    png_info = PngImagePlugin.PngInfo()
    meta = {
        "prompt": prompt,
        "negative": negative,
        "inference_steps": num_inference_steps,
        "prompt_fidelity": prompt_fidelity,
        "seed": seed,
    }
    parameters = json.dumps(meta)
    png_info.add_text("parameters", parameters)
    index = int(index)
    image = imgs[index]
    image_name = uuid.uuid4().hex
    Path(folder_images_path).mkdir(parents=True, exist_ok=True)
    image.save(f"{folder_images_path}/{image_name}.png", pnginfo=png_info)


def get_prompt_fidelities():
    return [3, 5, 8, 13, 21, 30]


def get_seed() -> int:
    return random.randrange(10 ** 8, 10 ** 10)


def get_image_sizes() -> List[ImageSize]:
    # The ratios are taken from Dreamstudio
    return [
        ImageSize(width=1024, ratio="1/1"),
        ImageSize(width=1152, height=896, ratio="5/4"),
        ImageSize(width=1216, height=832, ratio="3/2"),
        ImageSize(width=1536, height=640, ratio="21/9"),
        ImageSize(width=1344, height=768, ratio="16/9"),
        ImageSize(width=640, height=1536, ratio="9/21"),
        ImageSize(width=768, height=1344, ratio="9/16"),
    ]


def get_sdxl_models() -> Dict[str, SdxlModel]:
    return {
        "0.9": SdxlModel(
            version="0.9",
            base_model_path=constants.SDXL_BASE_0_9_MODEL_PATH,
            refiner_model_path=constants.SDXL_REFINER_0_9_MODEL_PATH,
        ),
        "1.0": SdxlModel(
            version="1.0",
            base_model_path=constants.SDXL_BASE_1_0_MODEL_PATH,
            refiner_model_path=constants.SDXL_REFINER_1_0_MODEL_PATH,
        ),
    }
