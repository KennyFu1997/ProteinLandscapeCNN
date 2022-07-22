import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm #进度条的库


def train(net, loss, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, optim='sgd', init=True, scheduler_type='Cosine'):
    
    # Initialize the net parameters.
    # Result: apply initialization to all conv2d and linear layers of "net".
    def init_xavier(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    if init:
        net.apply(init_xavier)

    # Print training device info and move net onto device.
    # Result: "net" to device.
    print('training on:', device)
    net.to(device)
    
    # Different optimizers.
    # Result: define "optimizer".
    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=0)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=0)
    elif optim == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=0)
    
    # Scheduler for learning rate.
    # Result: define "scheduler".
    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)
        
    # Training.
    # Result:
    train_losses = []
    train_acces = []
    eval_acces = []
    best_acc = 0.0
    for epoch in range(num_epoch):
        print("-----Epoch: {}-----".format(epoch + 1))
        # Switch to train.
        net.train()
        # Average traing accuracy among all batches is 0 at the begining.
        train_acc = 0
        for batch in tqdm(train_dataloader, desc='Training...'):
            imgs, targets = batch
            # Move inputs and targets onto device.
            imgs = imgs.to(device)
            targets = targets.to(device)
            # Get output.
            output = net(imgs)
            # Compute loss.
            Loss = loss(output, targets)
            # Clear optimizer's gradients.
            optimizer.zero_grad()
            # And compute gradient.
            Loss.backward()
            # One-time gradient descent.
            optimizer.step()
            # Compute training accuracy.
            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            acc = num_correct / batch_size
            train_acc += acc
        # Finished 1 epoch.
        # After each epoch, reset scheduler.
        scheduler.step()
        # Print current loss(after each epoch), and average accuracy among all batches.
        print("Epoch: {}, Loss: {}, Acc: {}".format(epoch, Loss.item(), train_acc / len(train_dataloader)))
        # Record average accuracy and current loss among all batches for each epoch.
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(Loss.item())
        
        # Switch to validation. (After 1 epoch, do validation)
        net.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for imgs, targets in valid_dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = net(imgs)
                Loss = loss(output, targets)
                _, pred = output.max(1)
                num_correct = (pred == targets).sum().item()
                # Accumulate loss and accuracy among all batches.
                eval_loss += Loss
                acc = num_correct / imgs.shape[0]
                eval_acc += acc
            # Average them.
            eval_losses = eval_loss / (len(valid_dataloader))
            eval_acc = eval_acc / (len(valid_dataloader))
            # If a better accuracy is got, save the check point.
            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(net.state_dict(),'best_acc.pth')
            # Record average accuracy among all batches for each epoch.
            eval_acces.append(eval_acc)
            # Print average loss and accuracy among all batches on the validation set after each epoch.
            print("Loss on validation set: {}".format(eval_losses))
            print("Accuracy on validation set: {}".format(eval_acc))
    # train_losses: average loss among all batches for each epoch.
    # train_acces: average accuracy among all batches for each epoch.
    # eval_acces: average accuracy among all batches for each epoch. (validation)
    return train_losses, train_acces, eval_acces
