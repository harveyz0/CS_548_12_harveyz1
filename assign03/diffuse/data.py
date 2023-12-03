from os import listdir, makedirs
from os.path import join
from logging import debug
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class Skulls(Dataset):
    def __init__(self, img_directory, transform=None, target_transform=None):
        self.img_directory = img_directory
        self.transform = transform
        self.target_transform = target_transform

        #We're going to sort the list so that we always have the same order every run.
        self.all_files = listdir(self.img_directory)
        self.all_files.sort()
        if 0 == len(self.all_files):
            raise FileNotFoundError(f'There were no files inside of directory {self.img_directory}')

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        path = join(self.img_directory, self.all_files[idx])
        image = read_image(path)
        if self.transform:
            image = self.transform(image)
        return image


def do_resize(source_directory, dest_directory, resize_to_height, resize_to_width):
    resize = Resize([resize_to_height, resize_to_width])
    makedirs(dest_directory, exist_ok=True)
    for img_path in listdir(source_directory):
        out_file = join(dest_directory, img_path)
        debug(f'Writing to file {out_file}')
        resize.forward(Image.open(join(source_directory, img_path))).save(out_file)


def skulls_dataset(directory="train_images/skull2dog", batch_size=64, shuffle=True):
    return DataLoader(Skulls(directory), batch_size=batch_size, shuffle=shuffle)
