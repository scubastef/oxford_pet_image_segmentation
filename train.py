from torch.utils.data import DataLoader
import torch
from dataset import OxfordPetDataset
from network import UNET
from transforms import transform

img_dir = '/Users/stefanswandel/PythonProjects/oxford-iiit-pet/images'
msk_dir = '/Users/stefanswandel/PythonProjects/oxford-iiit-pet/annotations/trimaps'


def mean_teacher_training(student : UNET, teacher : UNET, hyperparams : dict):

    # get train and val dataloaders
    traindataset = OxfordPetDataset(img_dir, msk_dir, transform=None, set_type='train', frac_labeled=0.1)
    trainloader = DataLoader(traindataset, batch_size=hyperparams['batch_size'])

    valdataset = OxfordPetDataset(img_dir, msk_dir, set_type='val')
    valloader = DataLoader(valdataset)

    student.train()
    teacher.train()
    for epoch in range(hyperparams['num_epochs']):

        for image, mask in trainloader:
            labeled = mask != 0

            # model inputs
            student_inputs = image # + eda TODO
            teacher_inputs = image # + edaprime TODO

            # model predictions
            student_preds = student(student_inputs)
            teacher_preds = teacher(teacher_inputs)

            # compute losses
            classification_loss = labeled # * TODO
            consistency_loss = 0 # TODO

    




if __name__ == '__main__':
    #mean_teacher_training()
    traindataset = OxfordPetDataset(img_dir, msk_dir, transform=transform, set_type='train', frac_labeled=1)
    trainloader = DataLoader(traindataset, batch_size=5)
    for x, y in trainloader:
        loss = torch.square(x - x).sum()
        print(loss)
        break


