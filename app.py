#!/usr/bin/env python3
import configparser
import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
from PIL import PngImagePlugin
from PIL.Image import Image

import constants
from image_size import ImageSize
from sdxl_model import SdxlModel
from sdxl_service import SdxlService

config = configparser.ConfigParser()


def get_sdxl_models() -> Dict[str, SdxlModel]:
    return {
        "0.9": SdxlModel(
            version="0.9",
            base_model_path=constants.SDXL_BASE_0_9_MODEL_PATH,
            refiner_model_path=constants.SDXL_REFINER_0_9_MODEL_PATH,
        ),
        "1.0": SdxlModel(
            version="1.0",
            base_model_path=constants.SDXL_BASE_1_0_MODEL_PATH,
            refiner_model_path=constants.SDXL_REFINER_1_0_MODEL_PATH,
        ),
    }


def __load_configuration() -> Tuple[str, str]:
    config.read(constants.CONFIG_FILE)
    folders_images = config['folders']['images']
    return folders_images, folders_images


def __save_configuration(save_image_path: str):
    if 'folders' not in config:
        config['folders'] = {}
    config['folders']['images'] = save_image_path
    with open(constants.CONFIG_FILE, 'w') as configfile:
        config.write(configfile)


def get_prompt_fidelities():
    return [3, 5, 8, 13, 21, 30]


def get_seed() -> int:
    return random.randrange(10 ** 8, 10 ** 10)


def get_image_sizes() -> List[ImageSize]:
    # The ratios are taken from Dreamstudio
    return [
        ImageSize(width=1024, ratio="1/1"),
        ImageSize(width=1152, height=896, ratio="5/4"),
        ImageSize(width=1216, height=832, ratio="3/2"),
        ImageSize(width=1536, height=640, ratio="21/9"),
        ImageSize(width=1344, height=768, ratio="16/9"),
        ImageSize(width=640, height=1536, ratio="9/21"),
        ImageSize(width=768, height=1344, ratio="9/16"),
    ]


def save_img(
        imgs: List[Image],
        index: int,
        prompt: str,
        negative: str,
        num_inference_steps: int,
        prompt_fidelity: float,
        seed: int,
        folder_images_path: str):
    png_info = PngImagePlugin.PngInfo()
    meta = {
        "prompt": prompt,
        "negative": negative,
        "inference_steps": num_inference_steps,
        "prompt_fidelity": prompt_fidelity,
        "seed": seed,
    }
    parameters = json.dumps(meta)
    png_info.add_text("parameters", parameters)
    index = int(index)
    image = imgs[index]
    image_name = uuid.uuid4().hex
    Path(folder_images_path).mkdir(parents=True, exist_ok=True)
    image.save(f"{folder_images_path}/{image_name}.png", pnginfo=png_info)


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
        gr.Textbox(constants.UI_PLACEHOLDER_LABEL)
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
        sdxl.queue()
        sdxl.launch(server_name=constants.SERVER_NAME, debug=True)
