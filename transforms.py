import albumentations as A

transform = A.Compose(
    [
    A.Resize(244, 244),
    A.RandomCrop(244, 244),
    A.Rotate(180, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Blur(3, p=0.5),
    #ToTensorV2()
    ]
)