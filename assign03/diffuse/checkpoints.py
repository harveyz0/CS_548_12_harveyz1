from os.path import exists
import torch



def load_checkpoints(checkpoint_directory, network, optimizer, lr_scheduler):
    if not exists(checkpoint_directory):
        raise FileNotFoundError(
            f'File of {checkpoint_directory} does not exist')
    checkpoint = torch.load(checkpoint_directory)
    return (checkpoint[EPOCH] + 1, network.load_state_dict(checkpoint[MODEL]),
            optimizer.load_state_dict(checkpoint[OPTIMIZER]),
            lr_scheduler.load_state_dict(checkpoint[LR_SCHEDULER]))


def save_checkpoints(epoch,
                     network,
                     optimizer,
                     lr_scheduler,
                     checkpoint_filename):
    return torch.save(
        {
            EPOCH: epoch,
            MODEL: network,
            OPTIMIZER: optimizer,
            LR_SCHEDULER: lr_scheduler
        }, checkpoint_filename)
