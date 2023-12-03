from dataclasses import dataclass
from json import load, dump, dumps
from logging import debug, info, error

from os.path import exists

_load_default_config = True
_default_config_file_path = './cfg.json'


@dataclass
class Configs:
    img_height: int = 2767
    img_width: int = 4455
    image_size = [2767, 4455]
    img_channels: int = 3

    #img_resize_height: int = 830
    #img_resize_width: int = 1336
    img_resize_height: int = 207
    img_resize_width: int = 334

    batch_size: int = 4
    eval_batch_size: int = 4
    buffer_size: int = 4

    output_channels: int = 3

    gradient_accumulation_steps = 1
    start_epoch = 0
    total_epochs = 100
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 20
    overwrite_output_dir = True
    seed = 0
    mixed_precision: str = 'fp16'

    save_at_epoch: int = 1000
    log_dir: str = "BasicGenTrain"
    log_dir_add_timestamp: bool = True
    log_with: str = 'tensorboard'
    checkpoint_directory: str = "checkpoints"
    checkpoint_filename: str = "checkpoints"
    checkpoint_prefix: str = 'ckpt'
    root_dir: str = "."

    data_directory: str = "train_images/skull2dog/resizedA"
    orig_data_directory: str = "train_images/skull2dog/trainA"

    shuffle: bool = True

    checkpoint_file_start: str = ""
    config_file_path: str = ""


def get_default_config():
    return Configs()


def load_config(file_path):
    cfg = None
    with open(file_path) as f:
        cfg = Configs(**load(f))
        cfg.config_file_path = file_path
    return cfg


def dump_config(config):
    return dumps(config.__dict__)


def save_config(config, file_path):
    with open(file_path, 'w') as f:
        dump(config.__dict__, f)
    return config
