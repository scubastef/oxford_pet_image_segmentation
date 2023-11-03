from network import *
import torch
from utils import *
from dataset import *
from visualization import *

if __name__=='__main__':
    train_set, val_set, test_set = generate_datasets(None, None, frac_unlabeled=0.4, transform=data_transforms)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=12)

    model = UNET()
    model.load_state_dict(torch.load('/Users/stefanswandel/Downloads/saved_teacher_model.pt', map_location=torch.device('cpu')))
    model.to('cpu')

    check_accuracy(test_loader, model, 'cpu')

    model.to('cpu')

    # visualize
    num_correct = 0
    num_pixels = 0
    model.eval()
        
    with torch.no_grad():
        for x,y,_,_ in test_loader:
            x = x.float().to('cpu')
            y = y.float().to('cpu') 
            preds = torch.sigmoid(model(x)).argmax(1)

            display_images(x, preds)
            break




