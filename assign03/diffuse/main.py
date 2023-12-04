from argparse import ArgumentParser
from logging import getLogger, DEBUG, ERROR
from os.path import join

getLogger().setLevel(level=DEBUG)
getLogger("tensorflow").setLevel(ERROR)
getLogger("accelerate.tracking").setLevel(ERROR)


def arg_parser(args):
    parser = ArgumentParser(
        prog=args[0], description='Module to build and run a pix2pix model')
    parser.add_argument('-c',
                        '--checkpoint',
                        action='store',
                        help='Start from this checkpoint file')
    parser.add_argument('-f',
                        '--config',
                        action='store',
                        help='Load the provided config file')
    parser.add_argument('-d',
                        '--dump-config',
                        action='store_true',
                        help='Dump the config to stdout')
    parser.add_argument(
        '-l',
        '--load-model',
        action='store',
        help=
        'Load a model from a directory and then try to push it through all the images passed in'
    )
    parser.add_argument(
        '-r',
        '--resize',
        action='store_true',
        help='Resize images from orig_data_directory to data_directory')
    parser.add_argument('-e',
                        '--eval',
                        action='store_true',
                        help='Run the eval functions')
    return parser.parse_args([] if len(args) == 1 else args[1:])


def dump_stdout_config(parser):
    from diffuse.configs import dump_config
    print(dump_config(load_config_arg(parser)))
    return 0


def load_config_arg(parser):
    from diffuse.configs import load_config, get_default_config
    cfg = None
    if parser.config:
        cfg = load_config(parser.config)
        cfg.config_file_path = parser.config
    else:
        cfg = get_default_config()
    if parser.checkpoint:
        cfg.checkpoint_file_start = parser.checkpoint
    cfg.checkpoint_directory = join(cfg.log_dir, cfg.checkpoint_directory)
    return cfg


def main_run(parser):
    from diffuse.runner import full_run
    config = load_config_arg(parser)
    full_run(config)


def resize_images(parser):
    from diffuse.data import do_resize
    cfg = load_config_arg(parser)
    do_resize(cfg.orig_data_directory, cfg.data_directory,
              cfg.img_resize_height, cfg.img_resize_width)


def load_model_arg(parser):
    from diffuse.runner import load_model
    cfg = load_config_arg(parser)
    cfg.model_path = parser.load_model
    load_model(cfg)


def main(*args):
    parser = arg_parser(args)
    if parser.dump_config:
        dump_stdout_config(parser)
    elif parser.resize:
        resize_images(parser)
    elif parser.load_model:
        load_model_arg(parser)
    else:
        main_run(parser)
