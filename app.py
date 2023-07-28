#!/usr/bin/env python3


import sys
import uuid
from typing import Tuple

import gradio as gr
from PIL import PngImagePlugin

import constants
from image_size import ImageSize
from sdxl_model import SdxlModel
from sdxl_service import SdxlService


def get_sdxl_models():
    return {
        "0.9": SdxlModel(
            version="0.9",
            base_model_path=constants.SDXL_BASE_0_9_MODEL_PATH,
            refiner_model_path=constants.SDXL_REFINER_0_9_MODEL_PATH
        ),
        "1.0": SdxlModel(
            version="1.0",
            base_model_path=constants.SDXL_BASE_1_0_MODEL_PATH,
            refiner_model_path=constants.SDXL_REFINER_1_0_MODEL_PATH
        )
    }


def get_image_sizes():
    image_sizes_vertical: Tuple[ImageSize] = tuple(
        ImageSize(width=1024, ratio=r) for r in ["1/1", "3/4", "2/3", "9/16"]
    )
    image_sizes_horizontal: Tuple[ImageSize] = tuple(
        ImageSize(height=1024, ratio=r) for r in ["4/3", "3/2", "16/9"]
    )
    return image_sizes_vertical + image_sizes_horizontal


def save_img(imgs, index, prompt):
    index = int(index)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text('parameters', prompt)
    image = imgs[index]
    image_name = uuid.uuid4().hex
    image.save(f"{image_name}.png", pnginfo=png_info)


def main():
    sdxl_models = get_sdxl_models()
    sd_service = SdxlService(sdxl_models["1.0"])

    image_sizes = get_image_sizes()

    def get_image_size_by_index(index):
        return image_sizes[index]

    with gr.Blocks(title=str(sdxl_models["1.0"])) as sdxl:
        image_size_var = gr.State(value=get_image_size_by_index(0))
        with gr.Row():
            positive_prompt = gr.Textbox(
                label=constants.UI_IMAGE_DESCRIPTION_TEXTBOX_LABEL)
            negative_prompt = gr.Textbox(
                label=constants.UI_IMAGE_NEGATIVE_DESCRIPTION_TEXTBOX_LABEL)
        with gr.Row():
            with gr.Column():
                size_index = gr.Radio(choices=[f"{image_size}" for image_size in image_sizes],
                                      value=f"{image_size_var}", label="Size", type="index")

            with gr.Column():
                total_results = gr.Slider(
                    2, 20, value=2, step=1, label=constants.UI_TOTAL_RESULTS_SLIDER_LABEL,
                    info=constants.UI_TOTAL_RESULTS_SLIDER_INFO)
                num_inference_steps = gr.Slider(
                    5, 100, value=10, step=1, label=constants.UI_INFERENCE_STEPS_SLIDER_LABEL,
                    info=constants.UI_INFERENCE_STEPS_SLIDER_INFO)

                render_btn = gr.Button(value=constants.UI_BUTTON_VALUE)
        size_index.change(fn=get_image_size_by_index, inputs=size_index, outputs=image_size_var)

        gallery = gr.Gallery(
            label="Images", show_label=False, elem_id="gallery", height="auto", columns=4
        )
        generated_images = gr.State()

        render_btn.click(
            fn=sd_service.apply,
            inputs=[
                positive_prompt,
                negative_prompt,
                image_size_var,
                num_inference_steps,
                total_results
            ],
            outputs=[gallery, generated_images]
        )
        save_image_btn = gr.Button("Save selected image")
        selected_index = gr.State()

        def on_select(evt: gr.SelectData):
            return evt.index

        gallery.select(on_select, None, selected_index)

        save_image_btn.click(save_img, [generated_images, selected_index, positive_prompt], None)

    sdxl.launch(server_name=constants.SERVER_NAME)


if __name__ == '__main__':
    sys.exit(main())
