from unittest import TestCase

import constants
from image_size import ImageSize
from sdxl_model import SdxlModel
from sdxl_service import SdxlService


class TestSdxlService(TestCase):
    def setUp(self) -> None:
        self.positive_prompt = "Cartoon"
        self.negative_prompt = "Astronaut playing guitar on the Moon"
        self.height = 1024
        self.width = 1024
        self.num_inference_steps = 10
        self.total_results = 2
        self.seed = 123456
        self.prompt_fidelity = 8.0
        self.image_size = ImageSize(width=1024)
        self.sdxl_model = SdxlModel(
            version="1.0",
            base_model_path=constants.SDXL_BASE_1_0_MODEL_PATH,
            refiner_model_path=constants.SDXL_REFINER_1_0_MODEL_PATH
        )

    def test_init(self):
        try:
            SdxlService(self.sdxl_model)
        except BaseException as err:
            self.fail(f"Error: {err}")

    def test_apply_total_results(self):
        sd_service = SdxlService(self.sdxl_model)
        images = sd_service.apply(
            self.positive_prompt,
            self.negative_prompt,
            self.image_size,
            self.num_inference_steps,
            self.prompt_fidelity,
            self.total_results,
        )
        self.assertEqual(self.total_results, len(images))
