import torch

def consistency_loss(student_out : torch.tensor, teacher_out : torch.tensor):
    return torch.square(student_out - teacher_out).sum()

