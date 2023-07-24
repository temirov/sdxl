#!/usr/bin/env python3


import sys

import gradio as gr

import constants
from sd_service import SdService


def main():
    sd_service = SdService()
    with gr.Blocks(title=constants.UI_PAGE_TITLE) as sdxl:
        positive_prompt = gr.Textbox(
            label=constants.UI_IMAGE_DESCRIPTION_TEXTBOX_LABEL)
        negative_prompt = gr.Textbox(
            label=constants.UI_IMAGE_NEGATIVE_DESCRIPTION_TEXTBOX_LABEL)
        total_results = gr.Slider(
            2, 20, value=2, step=1, label=constants.UI_TOTAL_RESULTS_SLIDER_LABEL,
            info=constants.UI_TOTAL_RESULTS_SLIDER_INFO)
        width = gr.Slider(
            1024, 2048, step=16, value=1024, label=constants.UI_IMAGE_WIDTH_SLIDER_LABEL,
            info=constants.UI_IMAGE_SIZE_SLIDER_INFO)
        # TODO: replace height with aspect ratio
        height = gr.Slider(
            1024, 2048, step=16, value=1024, label=constants.UI_IMAGE_HEIGHT_SLIDER_LABEL,
            info=constants.UI_IMAGE_SIZE_SLIDER_INFO)
        num_inference_steps = gr.Slider(
            5, 100, value=10, step=1, label=constants.UI_INFERENCE_STEPS_SLIDER_LABEL,
            info=constants.UI_INFERENCE_STEPS_SLIDER_INFO)

        render_btn = gr.Button(value=constants.UI_BUTTON_VALUE)
        render_btn.click(
            fn=sd_service.apply,
            inputs=[
                positive_prompt,
                negative_prompt,
                height,
                width,
                num_inference_steps,
                total_results
            ],
            outputs=gr.Gallery(
                label="Images", show_label=False, elem_id="gallery", height="auto", columns=4
            )
        )
        # TODO: Add an ability to save a file (include prompt in the PNG info)
    sdxl.launch(server_name='0.0.0.0')


if __name__ == '__main__':
    sys.exit(main())
