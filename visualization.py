import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def display_images(images, masks, figsize=(15,10), num_cols=6):

    images = (images).numpy().transpose((0,2,3,1))

    img = lambda x: Image.fromarray((x).astype('uint8'))
    images = [img(i) for i in images]

    masks = masks.unsqueeze(1)
    masks = (masks * 100).numpy().transpose((0, 2, 3, 1))
    msk = lambda x: Image.fromarray((x.squeeze()).astype('uint8'), 'L')
    masks = [msk(i) for i in masks]


    num_images = len(images) + len(masks)

    num_rows = int(np.ceil(num_images / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()

    img_idx = 0
    for i in range(0, num_images, 2):
        axs[i].imshow(images[img_idx])
        axs[i+1].imshow(masks[img_idx])

        axs[i].axis('off')
        axs[i+1].axis('off')

        img_idx += 1

    plt.show()