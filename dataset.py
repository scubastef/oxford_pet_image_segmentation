import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class OxfordPetDataset(Dataset):
    set_partitions = {'train': (0,5914), 'val' : (5914, 6284), 'test' : (6284, 7393)}

    def __init__(self, image_dir, mask_dir, transform = None, set_type = 'train', frac_labeled = 1) -> None:
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._transform = transform

        self._image_names = os.listdir(image_dir)[self.set_partitions[set_type][0] : self.set_partitions[set_type][1]]

        self._has_mask = None
        if set_type == 'train':
            self._has_mask = np.random.choice([False,True], len(self._image_names), p=[1-frac_labeled, frac_labeled])

    def __len__(self):
        return len(self._image_names)
    

    def __getitem__(self, index):
        image_path = os.path.join(self._image_dir, self._image_names[index])
        mask_path = os.path.join(self._mask_dir, self._image_names[index].replace('jpg', 'png'))

        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.float32)

        if self._transform is not None:
            augmentations = self._transform(image=image, mask=mask)
            image, mask = augmentations['image'], augmentations['mask']

        if self._has_mask is not None and not self._has_mask[index]:
            mask = 0

        
        return image, mask
    
if __name__ == '__main__':
    img_dir = '/Users/stefanswandel/PythonProjects/oxford-iiit-pet/images'
    msk_dir = '/Users/stefanswandel/PythonProjects/oxford-iiit-pet/annotations/trimaps'
    ds = OxfordPetDataset(img_dir, msk_dir, set_type='train', frac_labeled=0.5)
    i, m = ds.__getitem__(5)
    print(m)
