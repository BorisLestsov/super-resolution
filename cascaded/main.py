from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_test_set
import numpy

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

num_upscales = 3

print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor, num_scales=num_upscales)
test_set = get_test_set(opt.upscale_factor, num_scales=num_upscales)

print(train_set, '<- train set')
print(test_set, '<- test set')

training_data_loader = DataLoader(dataset=train_set, 
                                  num_workers=opt.threads, 
                                  batch_size=opt.batchSize, 
                                  shuffle=True)

testing_data_loader = DataLoader(dataset=test_set, 
                                 num_workers=opt.threads, 
                                 batch_size=opt.testBatchSize, 
                                 shuffle=False)

print('===> Building model')
#model = Net(upscale_factor=opt.upscale_factor)
model = torch.load("model_dump.pt")

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from skimage import io, transform

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h

        return [relu1_2, relu2_2, relu3_3, relu4_3]

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.model = Vgg16().eval().cuda()
        
    #def tot_size(self, x):
    #    tot = 1.0
    #    for i in x:
    #        tot += i
    #    return tot
    
    def features(self, x):
        return self.model.forward(x)
    
    def forward(self, x, y):
        feat_x = self.model.forward(x)
        feat_y = self.model.forward(y)
        feat_mse_sum = 0.0
        total_pixels = 0.0
        for i in range(len(feat_x)):
            feat_mse_sum += torch.sum((feat_x[i] - feat_y[i])**2)
            #total_pixels += self.tot_size(feat_x[i].size())
            total_pixels += feat_x[i].numel()
        return feat_mse_sum/total_pixels

def huber_function(inp, eps=1.0e-8):
    res = torch.sqrt(inp ** 2 + eps ** 2).mean()
    return res

def binary_loss(pred, targ, eps=1.0e-8):
    res = huber_function(pred - targ)
    return res

def unary_loss(pred):
    
    x_diff = pred[:, :, 1:, 1:] - pred[:, :, :-1, 1:]
    y_diff = pred[:, :, 1:, 1:] - pred[:, :, 1:, :-1]
    
    diff_norm = torch.sqrt(x_diff ** 2 + y_diff ** 2)
    res = huber_function(diff_norm)
    return res

vgg_loss = VGGLoss()

def total_loss(pred, target):
    res = 0
    for index in range(len(target)):
        one_pred = pred[index]
        one_targ = target[index]
        
        ul = 3.0 * unary_loss(one_pred)
        bl = 2.0 * binary_loss(one_pred, one_targ)
        vggl = 500.0 * vgg_loss(one_pred, one_targ)
        
        print(index, ul.data[0], bl.data[0], vggl.data[0], '<- losses')
        res = (
            ul + 
            bl + 
            vggl)
    return res


criterion = total_loss

if cuda:
    model = model.cuda()
    #criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1.0e-2)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        
        input, target = Variable(batch[0]), batch[1]
        if cuda:
            input = input.cuda()
            for target_index in range(len(target)):
                target[target_index] = Variable(target[target_index]).cuda()

        optimizer.zero_grad()
        loss = criterion(model(input, num_upscales=num_upscales), target)
        
        if not numpy.isnan(loss.data[0]):
            epoch_loss += loss.data[0]
            loss.backward()
            optimizer.step()
        else:
            print("Cought NAN in optimization.Skipping batch")
        
        if iteration % 500 == 0:
            torch.save(model, 'model_dump.pt')
            print("Dumping the model")

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss.data[0]))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    #test()
    checkpoint(epoch)
