from typing import Optional

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from image_size import ImageSize
from no_watermark import NoWatermark
from sdxl_model import SdxlModel


class SdxlService:
    def __init__(self, sdxl_model: SdxlModel):
        self.sdxl_model = sdxl_model
        self.__post_init()

    def __post_init(self):
        self.pipe = self.__get_base_model_pipe()
        self.refiner = self.__get_refiner_model_pipe()

    def __get_base_model_pipe(self):
        pipe = DiffusionPipeline.from_pretrained(
            self.sdxl_model.base_model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.watermark = NoWatermark()
        pipe.to("cuda")
        return pipe

    def __get_refiner_model_pipe(self):
        refiner = DiffusionPipeline.from_pretrained(
            self.sdxl_model.refiner_model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        )
        refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
        refiner.watermark = NoWatermark()
        refiner.to("cuda")
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
