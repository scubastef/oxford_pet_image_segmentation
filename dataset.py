import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OxfordPetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, frac_unlabeled=0, train_frac=0.8, transform=None):
        self.transform = transform
    
        root_dir = "./data/"
        test_set = torchvision.datasets.OxfordIIITPet(root_dir, "test", "segmentation", download=True)
        train_set = torchvision.datasets.OxfordIIITPet(root_dir, "trainval", "segmentation",)

        self.data = torch.utils.data.ConcatDataset([train_set, test_set])

        num_imgs = len(self.data) #len(self.images)
        self.num_unlabeled = int(num_imgs * frac_unlabeled)
        if self.num_unlabeled > 0:
            self.unlabled_idx = torch.randperm(num_imgs)[:self.num_unlabeled]

        test_size = int(num_imgs * (1 - train_frac))
        randperm = torch.randperm(num_imgs, generator=torch.Generator().manual_seed(np.random.randint(10000)))
        self.test_idx = randperm[:test_size]
        self.train_idx = randperm[test_size:]
    
    def get_test_idx(self):
        return self.test_idx.tolist()
    
    def get_train_idx(self):
        return self.train_idx.tolist()

    def __len__(self):
        #return len(self.images)
        return len(self.data)
    
    def __getitem__(self, index):

        image = np.array(self.data[index][0].convert('RGB'))
        mask = np.array(self.data[index][1].convert('L'), dtype=np.float32)
        mask = mask - 1

        # perform data augmentation if part of the training set
        train_samp = (self.train_idx==index).sum() > 0
        if self.transform is not None and train_samp:
            image_s = image + 75*np.random.rand()*np.random.normal(size=image.shape)
            augmentations_s = self.transform(image=image_s, mask=mask)
            image_s = augmentations_s['image']
            mask_s = augmentations_s['mask']

            image_t = image + 75*np.random.rand()*np.random.normal(size=image.shape)
            augmentations_t = self.transform(image=image_t, mask=mask)
            image_t = augmentations_t['image']
            mask_t = augmentations_t['mask']

            if self.num_unlabeled > 0 and (self.unlabled_idx==index).sum() > 0: 
                mask_s, mask_t = -78*torch.ones_like(mask_s), -78*torch.ones_like(mask_t)

            return image_s, mask_s, image_t, mask_t
        else:
        # if its part of test or val set then just resize and convert to tensor
            simple_tform = A.Compose([A.Resize(244,244), ToTensorV2()])
            augmentations = simple_tform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        # check if this should be unlabeled
        if self.num_unlabeled > 0 and (self.unlabled_idx==index).sum() > 0: 
            mask_s, mask_t = -78*torch.ones_like(mask), -78*torch.ones_like(mask)
        
        return image, mask, image, mask
    
    
def generate_datasets(image_dir, mask_dir, frac_unlabeled=0.5, train_frac=0.8, val_frac=0.2, transform=None):
    '''
    val_fran : how to split trainval set
    '''
    # get the oxford dataset
    dataset = OxfordPetDataset(image_dir, mask_dir, frac_unlabeled, train_frac, transform)

    # retrieve the test portion
    test_idx = dataset.get_test_idx()
    test_data = torch.utils.data.Subset(dataset, test_idx)
    
    # take the validation set from the test set
    val_dataset, test_dataset = torch.utils.data.random_split(test_data, [val_frac, 1-val_frac])

    # get train dataset
    train_idx = dataset.get_train_idx()
    train_dataset = torch.utils.data.Subset(dataset, train_idx)

    return train_dataset, val_dataset, test_dataset