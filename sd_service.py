from typing import Optional

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

import constants
from image_size import ImageSize
from no_watermark import NoWatermark


class SdService:
    def __init__(self):
        self.pipe = self.__get_base_model_pipe()
        self.refiner = self.__get_refiner_model_pipe()

    @staticmethod
    def __get_base_model_pipe():
        pipe = DiffusionPipeline.from_pretrained(
            constants.SDXL_BASE_0_9_MODEL_PATH,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.watermark = NoWatermark()
        return pipe

    @staticmethod
    def __get_refiner_model_pipe():
        refiner = DiffusionPipeline.from_pretrained(
            constants.SDXL_REFINER_0_9_MODEL_PATH,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        )
        refiner.enable_model_cpu_offload()
        refiner.watermark = NoWatermark()
        return refiner

    @staticmethod
    def __get_generator(seed: Optional[int]):
        if seed is None:
            generator = torch.Generator(device="cuda")
        else:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        return generator

    def __generate_image(
            self,
            generator,
            prompt,
            negative_prompt,
            height,
            width,
            num_inference_steps
    ):
        latent_images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            output_type="latent",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps
        ).images
        image = self.refiner(prompt=prompt, generator=generator, image=latent_images).images[0]
        return image

    def apply(
            self,
            positive_prompt: str,
            negative_prompt: str,
            image_size: ImageSize,
            num_inference_steps: int,
            total_results: int,
            seed: Optional[int] = None
    ):
        generator = self.__get_generator(seed)

        images = []
        for _ in range(total_results):
            image = self.__generate_image(
                generator,
                positive_prompt,
                negative_prompt,
                image_size.height,
                image_size.width,
                num_inference_steps
            )
            images.append(image)

        return images, images
