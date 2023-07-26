#!/usr/bin/env python3


import sys
from typing import Tuple

import gradio as gr

import constants
from image_size import ImageSize
from sd_service import SdService


def main():
    sd_service = SdService()
    image_sizes: Tuple[ImageSize] = tuple(ImageSize(width=1024, ratio=r) for r in ["1/1", "3/4", "2/3", "9/16"])

    def get_image_size_by_index(index):
        return image_sizes[index]

    with gr.Blocks(title=constants.UI_PAGE_TITLE) as sdxl:
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
        render_btn.click(
            fn=sd_service.apply,
            inputs=[
                positive_prompt,
                negative_prompt,
                image_size_var,
                num_inference_steps,
                total_results
            ],
            outputs=gr.Gallery(
                label="Images", show_label=False, elem_id="gallery", height="auto", columns=4
            )
        )
        # TODO: Add an ability to save a file (include prompt in the PNG info)
    sdxl.launch(server_name=constants.SERVER_NAME)


if __name__ == '__main__':
    sys.exit(main())
