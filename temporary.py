from utils import *
from dataset import *
import torch
from visualization import *
from network import *


if __name__=='__main__':
    train_set, val_set, test_set = generate_datasets(None, None, frac_unlabeled=0.4, transform=data_transforms)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=12)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=12)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=12)

    images, masks,_ , _ = next(iter(trainloader))

    display_images(images, masks)

    images, masks,_ , _ = next(iter(val_loader))

    display_images(images, masks)

    images, masks,_ , _ = next(iter(test_loader))

    display_images(images, masks)

    # model = UNET()
    # model.eval()

    # for image, mask, _, _ in test_loader:
    #     image = image.float()
    #     pred_mask = torch.sigmoid(model(image)).argmax(1)
    #     display_images(image, pred_mask)
    #     display_images(image, mask)
    #     break