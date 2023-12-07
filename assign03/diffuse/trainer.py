from os import makedirs
from os.path import join, exists
from logging import debug
import torch
from diffusers import UNet2DModel
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler
from accelerate import Accelerator, notebook_launcher
from diffusers.utils import make_image_grid
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.cuda import memory_stats
from pprint import pprint


class Trainer:
    MODEL = 'network'
    EPOCH = 'epoch'
    OPTIMIZER = 'optimizer'
    LR_SCHEDULER = 'lr_scheduler'

    def __init__(self,
                 image_size,
                 learning_rate,
                 num_warmup_steps,
                 total_epochs,
                 dataloader,
                 start_epoch=0,
                 checkpoint_file_start=None,
                 load_checkpoints=True):
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.model = self.get_model(image_size)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=learning_rate)
        self.total_epochs = total_epochs
        self.dataloader = dataloader
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=len(self.dataloader) * self.total_epochs)
        self.checkpoint_file_start = checkpoint_file_start
        self.start_epoch = start_epoch
        if load_checkpoints:
            self.load_checkpoints()

    @classmethod
    def from_config(cls, config, dataloader):
        return cls(config.image_size, config.learning_rate,
                   config.lr_warmup_steps, config.total_epochs, dataloader,
                   config.start_epoch, config.checkpoint_file_start)

    def build_accelerator(self,
                          gradient_accumulation_steps,
                          log_dir,
                          mixed_precision='fp16',
                          log_with='tensorboard'):
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=log_with,
            project_dir=log_dir)
        self.model, self.optimizer, self.dataloader, self.noise_scheduler, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader, self.noise_scheduler,
            self.lr_scheduler)
        if self.accelerator.is_main_process:
            makedirs(log_dir, exist_ok=True)
            self.accelerator.init_trackers('training')

    def load_checkpoints(self):
        if not self.checkpoint_file_start or not exists(self.checkpoint_file_start):
            debug(f'No checkpoint file value of : {self.checkpoint_file_start}')
            return
        debug(f'Loading checkpoint {self.checkpoint_file_start}')
        checkpoint = torch.load(self.checkpoint_file_start)
        self.start_epoch = checkpoint[Trainer.EPOCH] + 1
        self.model.load_state_dict(checkpoint[Trainer.MODEL])
        self.optimizer.load_state_dict(checkpoint[Trainer.OPTIMIZER])
        self.lr_scheduler.load_state_dict(checkpoint[Trainer.LR_SCHEDULER])

    def get_model(self, image_size):
        return UNet2DModel(in_channels=3,
                           out_channels=3,
                           sample_size=image_size,
                           layers_per_block=2,
                           block_out_channels=(128, 128, 256, 256, 512, 512),
                           down_block_types=[
                               "DownBlock2D", "DownBlock2D", "DownBlock2D",
                               "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
                           ],
                           up_block_types=[
                               "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                               "UpBlock2D", "UpBlock2D", "UpBlock2D"
                           ])

    def train_loop(self,
                   gradient_accumulation_steps,
                   log_dir,
                   checkpoint_directory,
                   checkpoint_prefix,
                   save_image_epochs,
                   save_model_epochs,
                   eval_batch_size,
                   seed,
                   mixed_precision='fp16',
                   log_with='tensorboard'):
        self.build_accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with='tensorboard',
            log_dir=log_dir)

        test_dir = join(log_dir, "samples")
        makedirs(test_dir, exist_ok=True)
        global_step = 0

        for epoch in range(self.start_epoch, self.total_epochs):
            progress_bar = tqdm(total=len(self.dataloader),
                                disable=not self.accelerator.is_main_process)
            progress_bar.set_description(f'Epoch {epoch}')

            for batch in self.dataloader:
                clean_images = batch["images"]
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps, (bs, ),
                    device=clean_images.device).long()
                noisy_images = self.noise_scheduler.add_noise(
                    clean_images, noise, timesteps)

                with self.accelerator.accumulate(self.model):
                    #pprint(memory_stats())
                    stuff = self.model(noisy_images, timesteps, return_dict=False)
                    loss = torch.nn.functional.mse_loss(stuff[0], noise)
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "step": global_step,
                    "lr": self.lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)
                global_step += 1

                if self.accelerator.is_main_process:
                    pipeline = DDPMPipeline(unet= \
                                            self.accelerator.unwrap_model(self.model),
                                            scheduler=self.noise_scheduler)

                    if (epoch + 1) % save_image_epochs == 0:
                        self.make_image_grid(self.evaluate(pipeline, eval_batch_size, seed), test_dir, epoch)

                    if (epoch + 1) % save_model_epochs == 0:
                        pipeline.save_pretrained(log_dir)
                        self.save_checkpoints(epoch, checkpoint_directory,
                                              checkpoint_prefix)

    def evaluate(self, pipeline, eval_batch_size, seed):
        return pipeline(batch_size=eval_batch_size, generator=torch.manual_seed(seed)).images

    def make_image_grid(self, images, test_dir, epoch):
        width = int(len(images) / 2)
        height = width
        if 0 != (len(images) % 2):
            height = width + 1
        make_image_grid(images, width, height).save(join(test_dir, 'Image_%04d.png' % epoch))

    def save_images(self, images, test_dir):
        i = 0 # This isn't gonna work right because it resets to zero every function call
        for img in images:
            img.save(join(test_dir, 'Image_%04d.png' % i))

    def save_checkpoints(self, epoch, checkpoint_directory, checkpoint_prefix):
        return torch.save(
            {
                Trainer.EPOCH:
                epoch,
                Trainer.MODEL:
                self.accelerator.unwrap_model(self.model).state_dict(),
                Trainer.OPTIMIZER:
                self.optimizer.state_dict(),
                Trainer.LR_SCHEDULER:
                self.lr_scheduler.state_dict()
            }, join(checkpoint_directory, checkpoint_prefix + f'-{epoch + 1}'))

    def launch_notebook(self,
                        gradient_accumulation_steps,
                        log_dir,
                        checkpoint_directory,
                        checkpoint_prefix,
                        save_image_epochs,
                        save_model_epochs,
                        eval_batch_size,
                        seed,
                        mixed_precision='fp16',
                        log_with='tensorboard',
                        num_processes=1):
        notebook_launcher(
            self.train_loop,
            (gradient_accumulation_steps, log_dir, checkpoint_directory,
             checkpoint_prefix, save_image_epochs, save_model_epochs,
             eval_batch_size, seed, mixed_precision, log_with),
            num_processes=num_processes)


def training_setup(config):
    pass
