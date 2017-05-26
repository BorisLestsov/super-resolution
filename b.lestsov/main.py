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
        
    def tot_size(self, x):
        tot = 1.0
        for i in x:
            tot += i
        return tot
    
    def features(self, x):
        return self.model.forward(x)
    
    def forward(self, x, y):
        feat_x = self.model.forward(x)
        feat_y = self.model.forward(y)
        feat_mse_sum = 0.0
        total_pixels = 0.0
        for i in range(len(feat_x)):
            feat_mse_sum += torch.sum((feat_x[i] - feat_y[i])**2)
            total_pixels += self.tot_size(feat_x[i].size())
        return feat_mse_sum#/total_pixels
        
    

def huber_loss(x, y, eps):
    return torch.sqrt((x-y)**2+eps) - torch.sqrt(eps)



def main():

    torch.cuda.set_device(1)
    a = VGGLoss()
    optimizer = optim.SGD(a.model.parameters(), lr=0.01)

    for i in range(10):
        a.model.parameters
        im1 = io.imread('/home/b.lestsov/datasets/test/super-res-test-div2/2012-05-27 19.15.35.jpg')
        im1 = transform.resize(im1, (224, 224))
        im1 = im1.swapaxes(0, 2)
        im1 = (im1-127.5)/128.0
        im1 = Variable(torch.Tensor(im1).unsqueeze(0).cuda(), requires_grad=False)
        
        im2 = io.imread('/home/b.lestsov/datasets/test/super-res-test-div2/2012-05-27 19.16.08.jpg')
        im2 = transform.resize(im2, (224, 224))
        im2 = im2.swapaxes(0, 2)
        im2 = (im2-127.5)/128.0
        im2 = Variable(torch.Tensor(im2).unsqueeze(0).cuda(), requires_grad=False)
        
        optimizer.zero_grad()
        vgg_loss = 
        loss = alpha*a.forward(im1, im2) + beta*huber()
        print(loss)
        loss.backward()
        optimizer.step()

if __name__=="__main__":
    main()


