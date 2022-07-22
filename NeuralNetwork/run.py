import torch

from dataset import *
from train import *
from dataloader import *
from getnet import *
from plot import *


if __name__ == '__main__':
    inp_path = "./inputs/micros.pt"
    lab_path = "./labels/labels.pt"
    train_scale = 0.9
    kernel_size = 5
    batch_size = 64
    num_epoch = 20
    lr = 0.1
    optim = 'sgd'
    lr_min = 1e-4
    
    train_dataloader, valid_dataloader = get_dataloader(inp_path, lab_path, train_scale, kernel_size, batch_size)
    device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
    net = ProtCNN()
    loss = nn.CrossEntropyLoss()
    train_losses, train_acces, eval_acces = train(net, loss, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, optim=optim)
    show_acces(train_losses, train_acces, eval_acces, num_epoch=num_epoch)

