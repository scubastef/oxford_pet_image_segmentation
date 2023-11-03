import torch
import torch.nn as nn
import torch.optim as optim
from dataset import *
from network import UNET
from utils import *
from visualization import display_images

# hyperparamers
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 14
NUM_EPOCHS = 23
NUM_WORKERS = 2
IMAGE_HEIGHT, IMAGE_WIDTH = 244, 244
FRAC_UNLABELED = 1
ALPHA = 0.999


global_step = 0

def train_fn(trainloader, valloader, student_model, teacher_model, optimiser, 
             class_loss_fn, con_loss_fn, scaler, alpha, epoch):
    
    global global_step
    student_model.train()
    teacher_model.train()
    for batch_idx, (s_data, s_targets, t_data, t_targets) in enumerate(trainloader):
        # move data to gpu
        s_data, t_data = s_data.float().to(DEVICE), t_data.float().to(DEVICE)
        s_targets, t_targets = s_targets.float().to(DEVICE), t_targets.float().to(DEVICE)
       
        # make student model preictions
        s_preds = student_model(s_data)

        # make teacher model predictions
        t_preds = teacher_model(t_data)

        # compute classification loss from student model; if label doesn't exist then loss is None
        class_loss = class_loss_fn(s_targets, s_preds)
    
        # compute consistency loss between student model and teacher model
        con_loss = con_loss_fn(s_preds, t_preds)

        # get total loss
        loss = class_loss + get_consistency_weight(epoch)*con_loss if class_loss is not None else con_loss
    
        # update student model weights using gradient descent
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        # update teacher weights using EMA
        if epoch > 0:
            with torch.no_grad():
                theta_prime = teacher_model.parameters()   # teacher params
                theta = student_model.parameters()         # student param

                for x,y in zip(theta_prime, theta):
                    a = min(1 - 1 / (global_step + 1), alpha)
                    x.copy_(a * x.data + (1 - a) * y.data)
        
            global_step += 1
        

        # print updates
        if batch_idx % 50 == 0:
            print(f'{batch_idx}')
            print('teacher:')
            check_accuracy(valloader, teacher_model, DEVICE)
            print('student')
            check_accuracy(valloader, student_model, DEVICE)
            print('\n')

def main():
    # define student and teacher models
    student_model = UNET().to(DEVICE)
    teacher_model = UNET().to(DEVICE)

    # generate student and teacher datasets
    train_dataset, val_dataset, test_dataset = generate_datasets(None, None, frac_unlabeled=FRAC_UNLABELED, 
                                                                 transform=data_transforms)

    # create dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # define optimiser
    optimiser = optim.Adam(student_model.parameters(), LEARNING_RATE)
        
    # define scaler
    scaler = torch.cuda.amp.GradScaler()

    # define loss
    def classification_loss(targets, preds, loss_fn = nn.CrossEntropyLoss()):
        loss = 0
        for i in range(len(targets)):
            if targets[i][1][1] != -78:
                loss += loss_fn(preds[i].unsqueeze(0).to(DEVICE), 
                                targets[i].unsqueeze(0).type(torch.LongTensor).to(DEVICE))
    
        if loss==0:
            return None
        else:
            return loss
    
    def consistency_loss(sp, tp):
        assert sp.size() == tp.size()
        input_softmax = torch.softmax(sp, dim=1)
        target_softmax = torch.softmax(tp, dim=1)
        return ((input_softmax - target_softmax)**2).mean()

    # train the model
    for epoch in range(NUM_EPOCHS):
        student_model.train()
        teacher_model.train()
        print(f'begining epoch {epoch + 1}\n')
        train_fn(trainloader, valloader, student_model, teacher_model, optimiser, classification_loss,
                  consistency_loss, scaler, ALPHA, epoch)
    

    # save the student and teacher model
    torch.save(student_model.state_dict(), '/content/gdrive/MyDrive/AppliedDL/saved_student_model_all_unlabeled.pt')
    torch.save(teacher_model.state_dict(), '/content/gdrive/MyDrive/AppliedDL/saved_teacher_model_all_unlabeled.pt')
    
    # test the model
    student_model.eval()
    teacher_model.eval()
    print('Teacher Model Test Accuracy:')
    check_accuracy(testloader, teacher_model, DEVICE)
    print('Student Model Test Accuracy:')
    check_accuracy(testloader, student_model, DEVICE)
    print('\n')
  

if __name__=='__main__':
    main()



