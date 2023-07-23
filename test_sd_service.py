from unittest import TestCase

from sd_service import SdService


class TestSdService(TestCase):
    def setUp(self) -> None:
        self.positive_prompt = "Cartoon"
        self.negative_prompt = "Astronaut playing guitar on the Moon"
        self.height = 1024
        self.width = 1024
        self.num_inference_steps = 10
        self.total_results = 2
        self.seed = 123456

    def test_init(self):
        try:
            SdService()
        except BaseException as err:
            self.fail(f"Error: {err}")

    def test_apply_total_results(self):
        sd_service = SdService()
        images = sd_service.apply(
            self.positive_prompt,
            self.negative_prompt,
            self.height,
            self.width,
            self.num_inference_steps,
            self.total_results,
        )
        self.assertEqual(self.total_results, len(images))
