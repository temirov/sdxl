from typing import List

import torch
from PIL.Image import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler, DiffusionPipeline, \
    AutoencoderKL, StableDiffusionXLControlNetPipeline

from c_n_model import CNModel
from canny_image_processor import CannyImageProcessor
from no_watermark import NoWatermark
from sdxl_model import SdxlModel


class ControlNetXLService:
    def __init__(self, sdxl_model: SdxlModel, control_net_model: CNModel):
        self.sdxl_model = sdxl_model
        self.cn_model = control_net_model
        self.__post_init()

    def __post_init(self):
        self.device = self.__get_device()
        self.pipe = self.__get_sd_model_pipe()
        self.refiner = self.__get_refiner_model_pipe(self.pipe)

    @staticmethod
    def __get_device() -> str:
        return "cuda"

    def __get_canny_model_pipe(self) -> ControlNetModel:
        return ControlNetModel.from_pretrained(
            self.cn_model.model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        )

    @staticmethod
    def __get_vae():
        return AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    def __get_sd_model_pipe(self):
        sd_model_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.sdxl_model.base_model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
            controlnet=self.__get_canny_model_pipe(),
            vae=self.__get_vae(),
        )
        sd_model_pipe.scheduler = UniPCMultistepScheduler.from_config(sd_model_pipe.scheduler.config)
        sd_model_pipe.watermark = NoWatermark()
        sd_model_pipe.enable_model_cpu_offload()
        return sd_model_pipe

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
        refiner.watermark = NoWatermark()
        refiner.enable_model_cpu_offload()
        return refiner

    def __get_generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(seed)

    def __generate_image(
            self,
            generator: torch.Generator,
            canny_image,
            canny_image_fidelity,
            prompt,
            negative_prompt,
            prompt_fidelity,
            num_inference_steps,
    ) -> Image:
        latent_images = self.pipe(
            prompt=prompt,
            image=canny_image,
            negative_prompt=negative_prompt,
            generator=generator,
            output_type="latent",
            num_inference_steps=num_inference_steps,
            guidance_scale=prompt_fidelity,
            controlnet_conditioning_scale=canny_image_fidelity
        ).images
        refined_images = self.refiner(
            prompt=prompt, generator=generator, image=latent_images, num_inference_steps=num_inference_steps,
        ).images
        return refined_images[0]

    def apply(
            self,
            positive_prompt: str,
            negative_prompt: str,
            canny_image: Image,
            canny_image_fidelity: float,
            num_inference_steps: int,
            prompt_fidelity: float,
            total_results: int,
            seed: int,
    ) -> List[Image]:
        generator = self.__get_generator(seed)
        prompt_fidelity_float = float(prompt_fidelity)
        canny_image = CannyImageProcessor(canny_image).apply()

        images = []
        for _ in range(total_results):
            image = self.__generate_image(
                generator,
                canny_image,
                canny_image_fidelity,
                positive_prompt,
                negative_prompt,
                prompt_fidelity_float,
                num_inference_steps,
            )
            images.append(image)

        return images
