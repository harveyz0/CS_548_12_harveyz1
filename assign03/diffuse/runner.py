from os import makedirs
from os.path import join
from diffusers import DDPMPipeline
from diffuse.data import skulls_dataset
from diffuse.trainer import Trainer
from pprint import pprint


def load_trainer(config):
    makedirs(config.log_dir, exist_ok=True)
    makedirs(config.checkpoint_directory, exist_ok=True)

    dataloader = skulls_dataset(config.data_directory, config.batch_size,
                                config.shuffle)
    return Trainer.from_config(config, dataloader)


def full_run(config):
    training = load_trainer(config)
    training.launch_notebook(config.gradient_accumulation_steps,
                             config.log_dir, config.checkpoint_directory,
                             config.checkpoint_prefix,
                             config.save_image_epochs,
                             config.save_model_epochs, config.eval_batch_size,
                             config.seed, config.mixed_precision,
                             config.log_with)


def load_model(config):
    to_dir = join(config.log_dir, config.generated_directory)
    makedirs(to_dir, exist_ok=True)
    training = load_trainer(config)
    pipeline = DDPMPipeline.from_pretrained(config.model_path)
    for i in range(100):
        training.evaluate(i, pipeline, to_dir, 10, 0)


def eval_generated(config):
    from torch_fidelity import calculate_metrics
    metrics = calculate_metrics(input2=config.data_directory,
                                input1=join(config.log_dir,
                                            config.generated_directory),
                                fid=True,
                                kid=True,
                                kid_subset_size=700,
                                cuda=True)
    print('All our metrics')
    pprint(metrics)
