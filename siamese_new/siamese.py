import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Dataset
import torchvision
import matplotlib.pyplot as plt


from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable


from torchvision import datasets, transforms
from torch import utils
from datasets import SiameseImageFolder

# Data preparation
CUDA = torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': False} if CUDA else {}
batch_size = 128
learning_rate = 1e-2
epochs = 50


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

dataset = datasets.ImageFolder('100_WebFace', transform=data_transform)
train_length = int(0.8 * len(dataset))
valid_length = len(dataset) - train_length
train, valid = utils.data.random_split(dataset=dataset, lengths=[train_length, valid_length])
train_dataset = SiameseImageFolder(train)
test_dataset = SiameseImageFolder(valid)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


# Network Buildup
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(), #32@24*24 32@108
                                     nn.MaxPool2d(2, stride=2), #32@12*12 32@54
                                     nn.Conv2d(32, 64, 5), nn.PReLU(), #64@8*8 32@50
                                     nn.MaxPool2d(2, stride=2)) #64*4*4 64@25

        self.fc = nn.Sequential(nn.Linear(64 * 25 * 25, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


# Define Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

embedding_net = EmbeddingNet()
model = SiameseNet(embedding_net)
if CUDA:
    model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, 25, gamma=0.5)
loss_function = ContrastiveLoss(1.)


# Start Training
for epoch in range(epochs):
    print('Epoch {}'.format(epoch))
    for step, (batch_x, batch_y) in enumerate(train_loader):
        if CUDA:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        embedding1, embedding2 = model(batch_x[0], batch_x[1])
        loss = loss_function(embedding1, embedding2, batch_y)
        # print(loss.grad_fn)
        optimizer.zero_grad()
        torch.autograd.backward(loss)
        optimizer.step()
        scheduler.step()
        print('Step {}, train loss {:.6f}'.format(step, loss.item()))

    test_loss = 0
    for batch_x, batch_y in test_loader:
        if CUDA:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        embedding1, embedding2 = model(batch_x[0], batch_x[1])
        test_loss += loss_function(embedding1, embedding2, batch_y)
    print('Step {}, test loss {:.6f}'.format(step, test_loss))