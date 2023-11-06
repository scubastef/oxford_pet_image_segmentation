from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from losses import consistency_loss, consistency_weight, classification_loss, cross_entorpy_loss
from dataset import OxfordPetDataset
from network import UNET
from transforms import train_transform, test_val_transform
from params import params


def mean_teacher_training(student : UNET, teacher : UNET, hyperparams : dict):
    img_dir, msk_dir = hyperparams['image_dir'], hyperparams['mask_dir']

    # get train and val dataloaders
    traindataset = OxfordPetDataset(img_dir, msk_dir, transform=train_transform, set_type='train', frac_labeled=0.5)
    trainloader = DataLoader(traindataset, batch_size=3)

    valdataset = OxfordPetDataset(img_dir, msk_dir, transform=test_val_transform, set_type='val')
    valloader = DataLoader(valdataset, batch_size=3)

    optimizer = Adam(student.parameters(), lr=hyperparams['lr'])

    student.train()
    teacher.train()
    for epoch in range(hyperparams['epochs']):

        for idx, (image, mask, labeled) in enumerate(trainloader):
            # model inputs
            student_inputs = image + torch.rand_like(image) # + eda TODO
            teacher_inputs = image + torch.rand_like(image) # + edaprime TODO

            # model predictions
            student_preds = student(student_inputs)
            teacher_preds = teacher(teacher_inputs)

            # compute losses
            class_loss = classification_loss(labeled, mask, student_preds)
            con_loss = consistency_weight(epoch, 80) * consistency_loss(student_preds, teacher_preds)
            total_loss = con_loss if class_loss is None else con_loss + class_loss

            # update student weights
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # update teacher weights
            alpha = min(1 - 1 / (epoch + 1), 0.999)
            with torch.no_grad():
                for s_param, t_param in zip(student.parameters(), teacher.parameters()):
                    t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data

        if True:#epoch % 1 == 0:
            
            student_loss, teacher_loss = 0, 0
            with torch.no_grad():
                for img, msk in valloader:
                    student_preds = student(img)
                    teacher_preds = teacher(img)

                    student_loss += cross_entorpy_loss(student_preds, msk)
                    teacher_loss += cross_entorpy_loss(teacher_preds, msk)
                
            print(f'Epoch {epoch} | Student Validation Loss: {student_loss / len(valdataset)} \
                  | Teacher Validation Loss: {teacher_loss / len(valdataset)}')



    




if __name__ == '__main__':
    student_model = UNET(3, 3)
    teacher_model = UNET(3, 3)
    mean_teacher_training(student_model, teacher_model, params)

