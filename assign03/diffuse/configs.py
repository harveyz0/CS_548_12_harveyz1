from dataclasses import dataclass, asdict
from json import load, dump, dumps
from logging import debug, info, error

from os.path import exists

_load_default_config = True
_default_config_file_path = './cfg.json'


@dataclass
class Configs:
    # Unused
    img_height: int = 2767
    img_width: int = 4455
    img_channels: int = 3
    buffer_size: int = 32
    log_dir_add_timestamp: bool = True
    # End Unused

    image_size = [64, 128]

    # Size of 830 by 1336 keeps the aspect ratio
    #img_resize_height: int = 830
    #img_resize_width: int = 1336
    img_resize_height: int = 64
    img_resize_width: int = 128

    batch_size: int = 32
    eval_batch_size: int = 10

    gradient_accumulation_steps: int = 1
    # You don't really need to configure start epoch
    # it'll be set when you load a checkpoint
    start_epoch: int = 0
    total_epochs: int = 100
    learning_rate: int = 1e-4
    lr_warmup_steps: int = 500
    # save_image_epochs will write out a grid of eval_batch_size big
    # Though this process is what takes the longest
    save_image_epochs: int = 5
    save_model_epochs: int = 5
    seed: int = 0
    mixed_precision: str = 'fp16'

    log_dir: str = "BasicGenTrain"
    log_with: str = 'tensorboard'
    checkpoint_directory: str = "checkpoints"
    checkpoint_filename: str = "checkpoints"
    checkpoint_prefix: str = 'ckpt'

    # data_directory is where the data you want the model to operate on
    data_directory: str = "../train_images/skull2dog/resizedA"
    # orig_data_directory is only used during the resize operations.
    # It holds the original data.
    orig_data_directory: str = "../train_images/skull2dog/trainA"

    # When you load-model this it will use this join(log_dir, generated_directory) to
    # put all the generated images. Images will be named counting upwards with a limit
    # of generate_n_images * eval_batch_size
    generated_directory: str = 'model-generated'
    generate_n_images: int = 100

    shuffle: bool = True

    # If a checkpoint is to be loaded its full file path should be put here.
    # The --checkpoint argument does this for you.
    checkpoint_file_start: str = ""
    # The directory containing the model. Largely this is usually just log_dir because
    # I don't know what files it actually needs. This will auto populate when using the
    # load-model argument.
    model_path: str = ""
    # The config file that was loaded when the program started.
    config_file_path: str = ""

    # I'm gonna add this so I can pick up generating where I left off because I have to turn the computer off now.
    start_generating_at: int = 0

    def __post_init__(self):
        self.image_size = [self.img_resize_height, self.img_resize_width]

    @classmethod
    def from_json_file(cls, file_path):
        cfg = None
        with open(file_path) as f:
            cfg = cls(**load(f))
            cfg.config_file_path = file_path
        return cfg


def get_default_config():
    return Configs()


def load_config(file_path):
    cfg = None
    with open(file_path) as f:
        cfg = Configs(**load(f))
        cfg.config_file_path = file_path
    return cfg


def dump_config(config):
    return dumps(asdict(config))


def save_config(config, file_path):
    with open(file_path, 'w') as f:
        dump(asdict(config), f)
    return config
