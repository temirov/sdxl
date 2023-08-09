import configparser
from typing import Tuple

import constants

config = configparser.ConfigParser()


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