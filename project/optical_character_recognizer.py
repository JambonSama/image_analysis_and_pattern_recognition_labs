# -*- coding: utf-8 -*-

######################################################################

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor

######################################################################


class DigitNet(nn.Module):
    """
    DigitNet class : This class is a network that takes a single 
    channel 28x28 grayscale image of a digit and returns the digit class.
    """

    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # pooling layers
        self.pool = nn.MaxPool2d(2)
        # fully connected layers
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)
        # training variables
        self.crit = nn.CrossEntropyLoss()
        self.opti = optim.Adam(self.parameters())

    def forward(self, x):
        """
        Applies the forward pass.
        """
        x = self.pool(F.celu(self.conv1(x)))
        x = self.pool(F.celu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.celu(self.fc1(x))
        x = F.celu(self.fc2(x))
        x = F.celu(self.fc3(x))
        return x

    def train_batch(self, batch):
        inputs, classes = batch
        self.opti.zero_grad()
        outputs = self(inputs)
        loss = self.crit(outputs, classes)
        loss.backward()
        self.opti.step()
        return loss.item()

    def train_epoch(self, train_loader):
        """
        Trains the network for one epoch on the DataLoader passed.
        Parameter :
            train_loader -- DataLoader on which to train
        """
        loss_array = []
        for batch in train_loader:
            loss = self.train_batch(batch)
            loss_array.append(loss)
        return loss_array

    def train_net(self, epoch_num, train_loader):
        """
        Trains the networks on the DataLoader passed for a certain
        number of epochs.
        Parameters :
            epoch_num -- number of epochs to train
            train_loader -- DataLoader on which to train
        """
        loss_array = []
        for _ in range(epoch_num):
            losses = self.train_epoch(train_loader)
            loss_array.append(losses)
        return loss_array

    def test_net(self, test_loader):
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, classes = data
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += classes.size(0)
                correct += (predicted == classes).sum().item()
        return correct/total

    def determine_epoch_num(self, train_loader, valid_loader, max_epoch_num=200):
        """
        Estimates the optimal number of training epochs for the 
        network in order to avoid both over and under fitting.
        Parameters :
            train_loader -- DataLoader used to train the network
            valid_loader -- DataLoader used to verify wheter the training starts to overfit
            max_epoch_num -- max number of epoch on which to train
        """
        perf = self.test_net(valid_loader)
        threshold = 20
        cummulative_overfit = 0
        for epoch_index in range(max_epoch_num):
            self.train_epoch(train_loader)
            new_perf = self.test_net(valid_loader)
            if new_perf < perf:
                cummulative_overfit += 1
                if cummulative_overfit > threshold:
                    break
            else:
                perf = new_perf
                cummulative_overfit = 0
        return epoch_index - threshold

    def _fd(self, img, N=None, method="cropped"):
        """
        _fd computes the  Fourier Descriptors 
        Parameters:
            img: the image to compute the afd on
            N: the number of fourrier descriptors to keep
            method: cropped or padded
        Returns: 
            Z: array of fourrier descriptors
        """
        # Converting from RGB to grayscale if necessary
        if len(img.shape) == 3:
            img = cv.cvtColor(src=img, code=cv.COLOR_RGB2GRAY)

        # Converting to binary image
        _, img = cv.threshold(src=img, thresh=0, maxval=1,
                              type=(cv.THRESH_BINARY | cv.THRESH_OTSU))
        [numrows, numcols] = img.shape

        # Extracting the contours
        contours, _ = cv.findContours(
            image=img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        contours = np.asarray(contours).squeeze()

        if len(contours.shape) == 1:
            i = np.argmax([len(c) for c in contours])
            contours = (contours[i][:, :]).squeeze()

        # Complex periodic signal out of the contours
        y = contours[:, 0]
        x = contours[:, 1]
        z = x + y*1j
        Nin = z.size

        # Assigning default arg
        if N is None:
            N = Nin

        # Processing to get the fft
        Z = np.fft.fft(z)

        # Magic to get the correct signal length
        if Nin < N:
            dst = img.copy()
            cv.resize(img, dst, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
            Z, Nin, _, _, _, _ = _fd(dst, N, method)
        elif Nin > N:
            i = math.ceil(N/2)

            if method == "cropped":
                Z = np.concatenate((Z[:i], Z[-i:]))
            elif method == "padded":
                Z[i:-i] = 0
            else:
                raise ValueError(f"Incorrect 'method' : {method}.")

        m = np.absolute(Z)
        phi = np.angle(Z)

        return Z, Nin, m, phi, numrows, numcols

    def _afd(self, img, N=None, method="cropped"):
        """
        _afd computes the adjusted Fourier Descriptors 
        (invariant by translation, rotation, and scaling).
        Parameters:
            img: the image to compute the afd on
            N: the number of fourrier descriptors to keep
            method: cropped or padded
        Returns: 
            m: array of fourrier descriptors (without the 0 and 1 fd)
        """

        Z, _, _, _, _, _ = self._fd(img, N, method)
        Z = Z/Z[1]
        Z = Z[2:-1]
        m = np.absolute(Z)
        return m

    def _getRatio(self, img):
        """
        Returns the height width ratio of the minimal surrounding rectangle arround 
        the object in the image.
        Parameters:
            image: the image to get the ratio from
        Returns:
            ratio: the object width/height ratio. Always >= 1
        """
        # Thresholding the image to binary
        _, threshholded_img = cv.threshold(
            src=img, thresh=0, maxval=255, type=(cv.THRESH_BINARY | cv.THRESH_OTSU))

        # find the object contours
        contour, _ = cv.findContours(
            image=threshholded_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

        # serach the minimum area rectangle around the object
        min_rect = cv.minAreaRect(contour[0])
        (_, _), (width, height), _ = min_rect

        # Get the ratio always >= 1
        if width >= height:
            ratio = width/height
        else:
            ratio = height/width

        return ratio

    def get_digit(self, image):
        """
        Returns the predicted number from the CNN.
        Parameters:
            image: 28 by 28 binary (1 and 0) matrix
        Returns: 
            digit: char of the detected value
        """
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        img = Image.fromarray(image)
        img = Tensor(t(img))
        img = torch.unsqueeze(img, 0)
        output = self(img)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

    def get_sign(self, image):
        """
        Returns the predicted sign from the mlp
        Parameters:
            image: 28 by 28 binary (1 and 0) matrix
        Returns: 
            sign: char of the detected sign
        """
        sign = ""

        # Start by counting the number of contours of the binarised image
        _, threshholded_img = cv.threshold(
            src=image, thresh=0, maxval=255, type=(cv.THRESH_BINARY | cv.THRESH_OTSU))

        num_shapes, _ = cv.connectedComponents(image=threshholded_img)
        
        # if 3 => division, 2 => equal, 1 => other
        if (num_shapes - 1) == 3:
            sign = "/"
        elif (num_shapes - 1) == 2:
            sign = "="
        else:
            # search the fourrier descriptor, the second one is used
            m = self._afd(image, N=5, method="cropped")
            # Calculate the ratio to detect the "-""
            ratio = self._getRatio(image)
            if ratio > 2:
                sign = "-"
            elif m[1] < 0.1:
                sign = "*"
            else:
                sign = "+"

        return sign
