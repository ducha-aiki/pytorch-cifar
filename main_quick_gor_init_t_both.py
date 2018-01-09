'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable
lr_decay = 0.1

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/gor_initckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
steps_lr = [50, 75, 90]
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
def adjust_learning_rate(optimizer):
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * lr_decay
    return

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print('Val accuracy: ' + str(acc))
    if acc > best_acc:
        print('Saving.., acc = ' + str(acc))
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/resnet18_quick_gor_init_t_both.t7')
        best_acc = acc

import torch.optim as optim
def cos_dist(anchor,positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""
    return torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0)
def gor_filter_loss(model):
    loss = None
    for param in model.parameters():
        sh = param.shape
        if len(sh) > 1:
            #print sh
            weights = param.view(param.size(0),-1)
            dist_matrix = cos_dist(weights.t(),weights.t())**2
            eye = torch.autograd.Variable(1.0 - torch.eye(dist_matrix.size(1))).cuda()
            dist_without_min_on_diag = eye * dist_matrix
            max_neg_a = torch.max(dist_without_min_on_diag,1)[0]#) + 1e-8)
            if loss is None:
                loss = max_neg_a.mean()
            else:
                loss+=  max_neg_a.mean()
            dist_matrix2 = cos_dist(weights,weights)**2
            eye2 = torch.autograd.Variable(1.0 - torch.eye(dist_matrix2.size(1))).cuda()
            dist_without_min_on_diag2 = eye2 * dist_matrix2
            max_neg_a2 = torch.max(dist_without_min_on_diag2,1)[0]#) + 1e-8)
            loss+=  max_neg_a2.mean()
    return loss
net.train()
optimizer1 = optim.SGD(net.parameters(), lr=0.01,momentum=0.9, dampening=0.9)
for i in range(10000):
    optimizer1.zero_grad()
    loss = gor_filter_loss(net)
    if i %100 == 0:
        print (loss)
    loss.backward()
    optimizer1.step()

for epoch in range(0,100):
    train(epoch)
    # update the optimizer learning rate
    if epoch in steps_lr:
        adjust_learning_rate(optimizer)
    test(epoch)