#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a script that trains the OCR's CNN for digit classification.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import optical_character_recognizer as ocr

######################################################################

torch.seed()

N = 60000*2
n = 12000*2

######################################################################

t_000 = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
t_180 = transforms.Compose([transforms.RandomRotation(
    (180., 180.)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

set1_000 = torchvision.datasets.MNIST(
    root='../data/mnist', train=True, download=True, transform=t_000)
set2_000 = torchvision.datasets.MNIST(
    root='../data/mnist', train=False, download=True, transform=t_000)
set1_180 = torchvision.datasets.MNIST(
    root='../data/mnist', train=True, download=True, transform=t_180)
set2_180 = torchvision.datasets.MNIST(
    root='../data/mnist', train=False, download=True, transform=t_180)

set1_000.data = set1_000.data[set1_000.targets != 9]
set1_000.targets = set1_000.targets[set1_000.targets != 9]

set1_180.data = set1_180.data[set1_180.targets != 9]
set1_180.targets = set1_180.targets[set1_180.targets != 9]

set2_000.data = set2_000.data[set2_000.targets != 9]
set2_000.targets = set2_000.targets[set2_000.targets != 9]

set2_180.data = set2_180.data[set2_180.targets != 9]
set2_180.targets = set2_180.targets[set2_180.targets != 9]

set1 = torch.utils.data.ConcatDataset([set1_000, set1_180])
set2 = torch.utils.data.ConcatDataset([set2_000, set2_180])

indices = torch.randperm(N)

epoch_set = torch.utils.data.Subset(set1, indices[n:])
valid_set = torch.utils.data.Subset(set1, indices[:n])
train_set = set1
test_set = set2

print("Data sets done.")

######################################################################

epoch_loader = torch.utils.data.DataLoader(
    epoch_set, batch_size=4, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=4, shuffle=True, num_workers=2)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=4, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=4, shuffle=True, num_workers=2)

print("Data loaders done.")

######################################################################

net = ocr.DigitNet()
epoch_num = net.determine_epoch_num(epoch_loader, valid_loader)

print(f"Optimal number of epochs : {epoch_num}")

######################################################################

net.train_net(epoch_num, train_loader)

print("CNN trained.")

######################################################################

accuracy = net.test_net(test_loader)

print(f"Accuracy : {accuracy}.")

######################################################################

PATH = "./mnist_net.cnn"
torch.save(net.state_dict(), PATH)

print("Done")

######################################################################
