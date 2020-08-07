import torch.utils.data as data
from abc import ABC, abstractmethod
import os

class cdataset(data.Dataset, ABC):
    def __init__(self, config):
        pass


    @abstractmethod
    def __len__(self):
        return 0


    @abstractmethod
    def __getitem__(self, index):
        pass


def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def list_dataset(dir_path):
    images = []
    assert os.path.isdir(dir_path), '%s is not a valid directory' % dir_path

    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images