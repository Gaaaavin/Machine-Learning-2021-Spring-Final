import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import json
import os
import torch.optim as optim
import torch
from torch import utils
import matplotlib.pyplot as plt
from vgg_net import vgg
from center_loss import CenterLoss

# Define parameters
dataset_pth = '1k_face'
batch_size = 32
model_name = "vgg16"
class_num = 1000
feature_num = 4096
lr_init = 1e-4
epoch_num = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
load_model = False
load_loss = False
model_path = 'vgg16Net.pth'
loss_path = 'center_loss.pth'

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    transforms.Resize((224, 224)),
])

# Load dataset
dataset = datasets.ImageFolder(dataset_pth, transform=data_transform)
train_length = int(0.8 * len(dataset))
valid_length = len(dataset) - train_length
train, valid = utils.data.random_split(dataset=dataset, lengths=[train_length, valid_length])
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(valid, batch_size=1, shuffle=True, **kwargs)
val_num = len(valid)

net = vgg(model_name=model_name, num_classes=feature_num, init_weights=True)
net.to(device)
if load_model:
    net.load_state_dict(torch.load(model_path))
cross_entropy = nn.CrossEntropyLoss()
center_loss = CenterLoss(num_classes=class_num, feat_dim=feature_num, use_gpu=device)
if load_loss:
    center_loss.load_state_dict(torch.load(loss_path))

net_optimizer = optim.Adam(net.parameters(), lr=lr_init)
loss_optimizer = optim.SGD(center_loss.parameters(), lr=0.5)

best_acc = 0.0
save_path = './{}Net.pth'.format(model_name)
epochs = []
train_losses = []
test_accu = []
file = open('model_log.txt', 'w')
file.write('start training\n')
file.close()


for epoch in range(epoch_num):
    # if epoch % 25 == 0:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        net_optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = center_loss(outputs, labels.to(device))
        loss_optimizer.zero_grad()
        loss.backward()
        net_optimizer.step()
        loss_optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        labels = list(dataset.class_to_idx.values())
        for step, val_data in enumerate(valid):
            score = np.array([0] * len(labels))
            label = val_data[1]
            output = net(val_data[0].to(device).unsqueeze(0))
            for i in range(len(labels)):
                target = torch.tensor(labels[i]).unsqueeze(0).to(device)
                score[i] = center_loss(output, target)
            predict_y = labels[score.argmin()]
            # val_images, val_labels = val_data
            # optimizer.zero_grad()
            # outputs = net(val_images.to(device))
            # predict_y = torch.max(outputs, dim=1)[1]
            acc += int(predict_y == label)
            rate = (step + 1) / len(valid)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtest accuracy: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, acc), end="")
        val_accurate = acc / len(valid)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            torch.save(center_loss.state_dict(), 'center_loss.pth')
        epochs.append(epoch + 1)
        train_losses.append(running_loss / step)
        test_accu.append(val_accurate)
        print('\n[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
        file = open('vgg_log.txt', 'a')
        file.write('[epoch %d] train_loss: %.3f  test_accuracy: %.3f\n' %
                   (epoch + 1, running_loss / step, val_accurate))
        file.close()
        torch.cuda.empty_cache()

print('Finished Training')
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('epoch')
ax1.set_ylabel('training loss', color=color)
ax1.plot(epochs, train_losses, color=color, label='training loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('test accuracy', color=color)  # we already handled the x-label with ax1
ax2.plot(epochs, test_accu, color=color, label='test accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
plt.savefig('vgg.png')
