import torch
import numpy as np

cross_entorpy_loss = torch.nn.CrossEntropyLoss()

def consistency_weight(epoch, rampup_length):
    T = min(epoch, rampup_length) / rampup_length
    return (rampup_length==0) * 1 + (rampup_length!=0) * np.exp(-5 * ((1 - T)**2))
    


def consistency_loss(student_out : torch.tensor, teacher_out : torch.tensor):
    return torch.square(student_out - teacher_out).sum()


def classification_loss(labeled_bool : list[bool], y : torch.tensor, y_hat : torch.tensor):

    if sum(labeled_bool) == 0:
        return
    
    y = y[labeled_bool]
    y_hat = y_hat[labeled_bool]
    return cross_entorpy_loss(y_hat, y)



