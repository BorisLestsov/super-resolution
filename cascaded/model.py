import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt



def _get_orthogonal_init_weights(weights):
    fan_out = weights.size(0)
    fan_in = weights.size(1) * weights.size(2) * weights.size(3)

    u, _, v = svd(normal(0.0, 1.0, (fan_out, fan_in)), full_matrices=False)

    if u.shape == (fan_out, fan_in):
        return torch.Tensor(u.reshape(weights.size()))
    else:
        return torch.Tensor(v.reshape(weights.size()))


class Net(nn.Module):
    def __init__(self, upscale_factor=2):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, (5, 5), (1, 1), (2, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2))
        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.bn4 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        
        self.bn_fin = nn.BatchNorm2d(32)
        self.conv_fin = nn.Conv2d(32, 3 * upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        self.upsample = nn.UpsamplingNearest2d(scale_factor=upscale_factor)

        self._initialize_weights()

    def forward(self, x, num_upscales=1):
        to_out = []
        
        for upsample_index in range(num_upscales):
            base = self.upsample(x)
        
            signal = self.relu(self.conv1(self.bn1(x)))
        
            residual = self.relu(self.conv2(self.bn2(signal)))
            signal = signal + residual
        
            residual = self.relu(self.conv3(self.bn3(signal)))
            signal = signal + residual
        
            residual = self.relu(self.conv4(self.bn4(signal)))
            signal = signal + residual
        
            correction = self.pixel_shuffle(self.conv_fin(self.bn_fin(residual)))
        
            result = base + correction
            
            to_out.append(result)
            x = result
            
        return to_out

    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight) * sqrt(2))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight) * sqrt(2))
        self.conv3.weight.data.copy_(_get_orthogonal_init_weights(self.conv3.weight) * sqrt(2))
        self.conv4.weight.data.copy_(_get_orthogonal_init_weights(self.conv4.weight))
