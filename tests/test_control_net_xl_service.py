from unittest import TestCase

from diffusers.utils import load_image

import constants
from c_n_model import CNModel
from control_net_xl_service import ControlNetXLService
from sdxl_model import SdxlModel


class TestControlNetXLService(TestCase):

    def setUp(self) -> None:
        self.positive_prompt = "Cartoon"
        self.negative_prompt = "Astronaut playing guitar on the Moon"
        self.num_inference_steps = 10
        self.total_results = 2
        self.seed = 123456
        self.pose_image = load_image("person.png")
        self.canny_image = load_image("landscape.png")
        self.pose_fidelity = 1.0
        self.canny_fidelity = 0.8
        self.prompt_fidelity = 8.0
        self.sdxl_model = SdxlModel(
            version="1.0",
            base_model_path=constants.SDXL_BASE_1_0_MODEL_PATH,
            refiner_model_path=constants.SDXL_REFINER_1_0_MODEL_PATH
        )
        self.cn_canny_model = CNModel(
            version="1.0",
            description="Canny",
            model_path=constants.CN_CANNY_1_0_MODEL_PATH
        )

    def test_apply_xl(self):
        control_net_service = ControlNetXLService(self.sdxl_model, self.cn_canny_model)
        images = control_net_service.apply(
            self.positive_prompt,
            self.negative_prompt,
            self.canny_image,
            self.canny_fidelity,
            self.num_inference_steps,
            self.prompt_fidelity,
            self.total_results,
            self.seed
        )
        self.assertEqual(self.total_results, len(images))

    def test_init(self):
        try:
            ControlNetXLService(self.sdxl_model, self.cn_canny_model)
        except BaseException as err:
            self.fail(f"Error: {err}")
