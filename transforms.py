import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose(
    [
    A.Resize(244, 244),
    A.RandomCrop(244, 244),
    A.Rotate(180, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #A.Normalize(),
    ToTensorV2()
    ]
)

test_val_transform = A.Compose(
    [
    A.Resize(244, 244),
    #A.Normalize(),
    ToTensorV2()
    ]
)