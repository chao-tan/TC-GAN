import os.path
from datasets.cdataset import cdataset, list_dataset
import torchvision.transforms as transforms
from PIL import Image
import random


def unaligned_transform(config):
    transform_list = []
    osize = [int(config['img_size']), int(config['img_size'])]

    if config['status'] == 'train':
        crop_size = int(int(config['img_size'])*float(config['crop_scale']))
        transform_list.append(transforms.Resize([crop_size,crop_size], Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(osize))
        if config['flip'] is True:
            transform_list.append(transforms.RandomHorizontalFlip())
        if config['add_colorjit'] is True:
            transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.25, saturation=0.25, hue=0))
    elif config['status'] == 'test':
        transform_list.append(transforms.Resize(osize,Image.BICUBIC))
    else:
        pass

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class unalignedDataset(cdataset):
    def __init__(self, config):
        cdataset.__init__(self, config)

        self.A_paths = sorted(list_dataset(os.path.join(config['dataroot'], str.upper(config['status']) + '_A')))
        self.B_paths = sorted(list_dataset(os.path.join(config['dataroot'], str.upper(config['status']) + '_B')))

        self.test_len = len(self.A_paths)
        self.transform = unaligned_transform(config)
        self.config = config

    def __getitem__(self, index):
        A_path = self.A_paths[index % len(self.A_paths)]
        B_path = self.B_paths[(index+random.sample(range(len(self.B_paths)),1)[0]) % len(self.B_paths)]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        A_img, B_img = self.transform(A_img), self.transform(B_img)
        return {'A': A_img, "B": B_img,
                'PATH':A_path.replace(os.path.join(self.config['dataroot'], str.upper(self.config['status'])+'_A\\'),"")}

    def __len__(self):
        return self.test_len




