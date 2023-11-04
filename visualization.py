import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from dataset import OxfordPetDataset
from transforms import transform


def visualize_dataset(dataset : Dataset):

    fig, ax = plt.subplots(2, 6, figsize=(15,10), layout='tight')

    sample_idxs = np.random.randint(len(dataset), size=6)

    for i, idx in enumerate(sample_idxs):
        img, msk = dataset[idx]

        img = Image.fromarray(img.astype('uint8'))
        ax[0,i].imshow(img)
        ax[0,i].axis('off')

        msk = Image.fromarray(msk.astype('uint8'))
        ax[1,i].imshow(msk)
        ax[1,i].axis('off')
    
    plt.show()



if __name__ == '__main__':
    img_dir = '/Users/stefanswandel/PythonProjects/oxford-iiit-pet/images'
    msk_dir = '/Users/stefanswandel/PythonProjects/oxford-iiit-pet/annotations/trimaps'
    ds = OxfordPetDataset(img_dir, msk_dir, transform=transform, set_type='val')
    visualize_dataset(ds)