import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class OxfordPetDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform = None) -> None:
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._transform = transform

        self._image_names = os.listdir(image_dir)

    def __len__(self):
        return len(self._image_names)
    

    def __getitem__(self, index):
        image_path = os.path.join(self._image_dir, self._image_names[index])
        mask_path = os.path.join(self._mask_dir, self._image_names[index].replace('jpg', 'png'))

        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert('RGB'), dtype=np.float32)

        if self._transform is not None:
            augmentations = self._transform(image=image, mask=mask)
            image, mask = augmentations['image'], augmentations['mask']
        
        return image, mask