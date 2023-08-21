#!/usr/bin/env python3

import gradio as gr

import constants
from app_config import __save_configuration, __load_configuration
from app_helper import save_img, get_sdxl_models, get_image_sizes, get_prompt_fidelities, get_seed, load_images
from control_net_xl_service import ControlNetXLService
from image_size import ImageSize
from sdxl_service import SdxlService

sdxl_models = get_sdxl_models()
sd_service = SdxlService(sdxl_models["1.0"])

image_sizes = get_image_sizes()
prompt_fidelities = get_prompt_fidelities()

with gr.Blocks(title=str(sdxl_models["1.0"]), theme=gr.themes.Soft()) as sdxl:
    folder_images_path = gr.State()

    with gr.Tab(constants.UI_CREATE_TAB):
        with gr.Row():
            positive_prompt = gr.Textbox(label=constants.UI_IMAGE_DESCRIPTION_TEXTBOX_LABEL)
            negative_prompt = gr.Textbox(
                label=constants.UI_IMAGE_NEGATIVE_DESCRIPTION_TEXTBOX_LABEL
            )
        with gr.Row():
            with gr.Column():
                size_index = gr.Radio(
                    choices=[str(image_size) for image_size in image_sizes],
                    value=f"{image_sizes[0]}",
                    label="Size",
                    type="index",
                )
                total_results = gr.Slider(
                    2,
                    20,
                    value=2,
                    step=1,
                    label=constants.UI_TOTAL_RESULTS_SLIDER_LABEL,
                    info=constants.UI_TOTAL_RESULTS_SLIDER_INFO,
                )
            with gr.Column():
                num_inference_steps = gr.Slider(
                    5,
                    100,
                    value=10,
                    step=1,
                    label=constants.UI_INFERENCE_STEPS_SLIDER_LABEL,
                    info=constants.UI_INFERENCE_STEPS_SLIDER_INFO,
                )

                prompt_fidelity = gr.Radio(
                    choices=[f"{prompt_fidelity}" for prompt_fidelity in prompt_fidelities],
                    value=f"{prompt_fidelities[2]}",
                    label="Prompt fidelity",
                    type="value",
                )
                seed = gr.Number(value=get_seed, precision=0, label=constants.UI_SEED_LABEL)

                render_btn = gr.Button(value=constants.UI_GENERATE_BUTTON_VALUE)
                render_btn.click(fn=get_seed, inputs=None, outputs=seed)


        def get_image_size(index: str) -> ImageSize:
            index = int(index)
            return image_sizes[index]


        image_size_cur = gr.State(value=image_sizes[0])

        size_index.input(
            fn=get_image_size, inputs=[size_index], outputs=image_size_cur
        )

        gallery = gr.Gallery(
            show_label=False,
            height="auto",
            columns=4,
        )
        generated_images = gr.State()

        render_btn.click(
            fn=sd_service.apply,
            inputs=[
                positive_prompt,
                negative_prompt,
                image_size_cur,
                num_inference_steps,
                prompt_fidelity,
                total_results,
                seed
            ],
            outputs=[gallery, generated_images],
        )

        save_image_btn = gr.Button("Save selected image")
        selected_index = gr.State()


        def get_select_index(evt: gr.SelectData):
            return evt.index


        gallery.select(get_select_index, None, selected_index)

    with gr.Tab(constants.UI_PROCESS_TAB):
        gr.Markdown(constants.PROCESSING_HEADER)
        control_net_service_service = ControlNetXLService(
            sdxl_models["1.0"],
            sdxl_models["canny"]
        )
        with gr.Row():
            with gr.Column():
                positive_prompt = gr.Textbox(label=constants.UI_IMAGE_DESCRIPTION_TEXTBOX_LABEL)
                negative_prompt = gr.Textbox(
                    label=constants.UI_IMAGE_NEGATIVE_DESCRIPTION_TEXTBOX_LABEL
                )
                num_inference_steps = gr.Slider(
                    5,
                    100,
                    value=10,
                    step=1,
                    label=constants.UI_INFERENCE_STEPS_SLIDER_LABEL,
                    info=constants.UI_INFERENCE_STEPS_SLIDER_INFO,
                )
                total_results = gr.Slider(
                    2,
                    20,
                    value=2,
                    step=1,
                    label=constants.UI_TOTAL_RESULTS_SLIDER_LABEL,
                    info=constants.UI_TOTAL_RESULTS_SLIDER_INFO,
                )
                prompt_fidelity = gr.Radio(
                    choices=[f"{prompt_fidelity}" for prompt_fidelity in prompt_fidelities],
                    value=f"{prompt_fidelities[2]}",
                    label="Prompt fidelity",
                    type="value",
                )
                seed = gr.Number(value=get_seed, precision=0, label=constants.UI_SEED_LABEL)
            with gr.Column():
                canny_image = gr.Image(source='upload', type="numpy")
                canny_image_fidelity = gr.Slider(label="Fidelity", minimum=0.1, maximum=1.0, value=0.5, step=0.1)
                run_button = gr.Button(label="Run")
        with gr.Row():
            processing_gallery = gr.Gallery(
                show_label=False,
                height="auto",
                columns=4
            )

        run_button.click(
            fn=control_net_service_service.apply,
            inputs=[
                positive_prompt,
                negative_prompt,
                canny_image,
                canny_image_fidelity,
                num_inference_steps,
                prompt_fidelity,
                total_results,
                seed
            ],
            outputs=[processing_gallery]
        )
    with gr.Tab(constants.UI_GALLERY_TAB) as gallery_tab:
        saved_images = gr.Gallery(
            show_label=False,
            height="auto",
            columns=4,
        )
        gallery_images = gr.State()
        gallery_tab.select(load_images, [folder_images_path], [saved_images, gallery_images])
    with gr.Tab(constants.UI_SETTINGS_TAB):
        folders_images_textbox = gr.Textbox(label=constants.UI_SAVE_IMAGE_PATH_LABEL)
        save_configuration_btn = gr.Button(constants.UI_SAVE_CONFIGURATION_BUTTON_VALUE)
        save_configuration_btn.click(__save_configuration, [folders_images_textbox], None)
        save_configuration_btn.click(
            fn=lambda path: path, inputs=[folders_images_textbox], outputs=[folder_images_path]
        )

    sdxl.load(__load_configuration, None, [folders_images_textbox, folder_images_path])

    save_image_btn.click(
        save_img, [generated_images, selected_index, positive_prompt, negative_prompt, num_inference_steps,
                   prompt_fidelity, seed, folder_images_path], None
    )
    if __name__ == "__main__":
        sdxl.queue()  # .queue() switches system to WebSockets which seem to be better supported
        sdxl.launch(server_name=constants.SERVER_NAME, debug=True)
