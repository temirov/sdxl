from typing import List, Tuple

import torch
from PIL.Image import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from image_size import ImageSize
from no_watermark import NoWatermark
from sdxl_model import SdxlModel


class SdxlService:
    def __init__(self, sdxl_model: SdxlModel):
        self.sdxl_model = sdxl_model
        self.__post_init()

    def __post_init(self):
        self.device = self.__get_device()
        self.pipe = self.__get_base_model_pipe()
        self.refiner = self.__get_refiner_model_pipe(self.pipe)

    @staticmethod
    def __get_device():
        return "cuda"

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
        pipe.to(self.device)
        return pipe

    def __get_refiner_model_pipe(self, base_model):
        refiner = DiffusionPipeline.from_pretrained(
            self.sdxl_model.refiner_model_path,
            text_encoder_2=base_model.text_encoder_2,
            vae=base_model.vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        )
        refiner.unet = torch.compile(
            refiner.unet, mode="reduce-overhead", fullgraph=True
        )
        refiner.watermark = NoWatermark()
        refiner.to(self.device)
        return refiner

    def __get_generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(seed)

    def __generate_image(
            self,
            generator,
            prompt,
            negative_prompt,
            height,
            width,
            prompt_fidelity,
            num_inference_steps,
    ):
        latent_images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            output_type="latent",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=prompt_fidelity,
        ).images
        image = self.refiner(
            prompt=prompt, generator=generator, image=latent_images, num_inference_steps=num_inference_steps,
        ).images[0]
        return image

    def apply(
            self,
            positive_prompt: str,
            negative_prompt: str,
            image_size: ImageSize,
            num_inference_steps: int,
            prompt_fidelity: float,
            total_results: int,
            seed: int,
    ) -> Tuple[List[Image], List[Image]]:
        generator = self.__get_generator(seed)
        prompt_fidelity_float = float(prompt_fidelity)

        images = []
        for _ in range(total_results):
            image = self.__generate_image(
                generator,
                positive_prompt,
                negative_prompt,
                image_size.height,
                image_size.width,
                prompt_fidelity_float,
                num_inference_steps,
            )
            images.append(image)

        return images, images
