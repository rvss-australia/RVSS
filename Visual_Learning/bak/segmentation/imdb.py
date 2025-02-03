import os
import io
import h5py

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


class IMDB(Dataset):
    def __init__(self, root_dir, mode):
        super(IMDB, self).__init__()
        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [
            transforms.ColorJitter(brightness=0.4, contrast=0.2,
                                    saturation=0.4, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]
        )

        self.catalog_path = os.path.join(root_dir, mode + '.hdf5')
        try:
            os.path.exists(self.catalog_path)
        except FileExistsError:
            print('catalog does not exist')
        dataset = h5py.File(self.catalog_path, 'r')
        self.n_samples = len(dataset['images'])
        dataset.close()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return self.n_samples


class FruitIMDB(IMDB):
    def __init__(self, root_dir, mode):
        super(FruitIMDB, self).__init__(root_dir, mode)
    
    def __getitem__(self, idx):
        if not hasattr(self, 'dataset'):
            self.dataset = h5py.File(self.catalog_path, 'r')
        image = Image.open(io.BytesIO(self.dataset['images'][idx]))
        width, height = 256, 192
        image = image.resize((width, height))
        label = Image.open(io.BytesIO(self.dataset['labels'][idx]))
        reduction = 2
        label = label.resize((int(width/reduction),
                                int(height/reduction)), Image.NEAREST)
        angle = random.randint(-45, 45)
        if random.random() > 0.5:
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)
        if self.transform:
            image = self.transform(image)
        return image, np.array(label).astype(np.long)


class FaceIMDB(IMDB):
    def __init__(self, root_dir, mode):
        super(FaceIMDB, self).__init__(root_dir, mode)

    def __getitem__(self, idx):
        if not hasattr(self, 'dataset'):
            self.dataset = h5py.File(self.catalog_path, 'r')
        image = Image.open(io.BytesIO(self.dataset['images'][idx]))
        width, height = 256, 256
        image = image.resize((width, height))
        label = Image.open(io.BytesIO(self.dataset['labels'][idx]))
        reduction = 2
        label = label.resize((int(width/reduction),
                              int(height/reduction)), Image.NEAREST)
        angle = random.randint(-45, 45)
        if random.random() > 0.5:
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)
        if self.transform:
            image = self.transform(image)
        label = np.array(label)
        label = np.argmax(label, axis=2)
        return image, label
