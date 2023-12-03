from diffuse.data import skulls_dataset
from diffuse.trainer import Trainer


def full_run(config):
    dataloader = skulls_dataset(config.data_directory, config.batch_size,
                                config.shuffle)
    training = Trainer.from_config(config, dataloader)
    training.launch_notebook(config.gradient_accumulation_steps,
                             config.log_dir, config.checkpoint_directory,
                             config.checkpoint_prefix,
                             config.save_image_epochs,
                             config.save_model_epochs, config.eval_batch_size,
                             config.seed, config.mixed_precision,
                             config.log_with)
