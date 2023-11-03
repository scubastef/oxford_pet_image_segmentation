import torch 
import numpy as np
import torchvision
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import OxfordPetDataset
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_transforms(image, mask):
    transform = A.Compose(
        [
        A.Resize(244, 244),
        A.RandomCrop(244, 244),
        A.Rotate(180, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
        ]
    )

    return transform(image=image, mask=mask)



def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    model.eval()
        
    with torch.no_grad():
        for x,y,_,_ in loader:
            x = x.float().to(device)
            y = y.float().to(device) 
            preds = torch.sigmoid(model(x)).argmax(1)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(f'got {num_correct}/{num_pixels} with acc {num_correct/num_pixels * 100:.2f}')




def get_consistency_weight(epoch, rampup_length=10):
    if rampup_length == 0:
        return 1.0
    else:
        epoch = np.clip(epoch, 0.0, rampup_length)
        phase = 1.0 - epoch / rampup_length
        return float(np.exp(-5.0 * phase * phase))

