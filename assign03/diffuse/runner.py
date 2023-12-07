from os import makedirs, listdir
from os.path import join
from diffusers import DDPMPipeline
from diffuse.data import skulls_dataset, class_dataset
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
    pipeline = DDPMPipeline.from_pretrained(config.model_path).to("cuda")
    for i in range(config.generate_n_images):
        images = training.evaluate(pipeline, 10, i)
        for img in images:
            img.save(join(to_dir, 'generated_%04d.png' % i))


def eval_generated(config):
    # TODO Fix kid_subset_size
    from torch_fidelity import calculate_metrics
    num_files = len(listdir(join(config.log_dir, config.generated_directory)))
    metrics = calculate_metrics(input2=config.data_directory,
                                input1=join(config.log_dir,
                                            config.generated_directory),
                                fid=True,
                                kid=True,
                                kid_subset_size=num_files,
                                cuda=True)
    print('All our metrics')
    pprint(metrics)
