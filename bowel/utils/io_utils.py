import pathlib

import yaml


def load_config(config_path: pathlib.Path):
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


def save_yaml(data: dict[str, float], path: pathlib.Path):
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile)
