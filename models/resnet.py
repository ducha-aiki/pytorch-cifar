'''ResNet in PyTorch.

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
from torch.autograd import Variable
from torch.autograd.function import InplaceFunction
class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class RBFConv2d(nn.Module):
    def __init__(self, *args, **argv):
        super(RBFConv2d, self).__init__()
        self.conv = nn.Conv2d(*args,**argv)
        self.conv_ones = nn.Conv2d(self.conv.in_channels,1,self.conv.kernel_size,self.conv.stride,self.conv.padding)
        self.conv_ones.weight.data.fill_(1)
        self.conv_ones.bias.data.fill_(0)
       # self.conv1d = nn.Conv2d(self.conv.out_channels, self.conv.out_channels,1,1)
    def forward(self, x):
        self.conv_ones.weight.data.fill_(1)
        self.conv_ones.bias.data.fill_(0)
        dot_product = self.conv.forward(x)
        dot_product = dot_product - self.conv.bias.view(1,-1,1,1).expand_as(dot_product)
        weights_sq = ((self.conv.weight).norm(2,dim = 1).norm(2,dim = 2).norm(2,dim = 3)**2).view(1,-1,1,1)
        input_sq = self.conv_ones(x**2).detach() # stop gradiend
        distance_sq = weights_sq.expand_as(dot_product) - 2*dot_product + input_sq.expand_as(dot_product) + 1e-8
        return torch.exp(-distance_sq/16.0)
        #return self.conv1d(torch.exp(-distance_sq/16.0))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNetRBFCLF(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, rbf = False, rbfLast = False):
        super(ResNetRBFCLF, self).__init__()
        self.in_planes = 64
        self.dorbflast = rbfLast 
        self.features = nn.Sequential(
            conv3x3(3,64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(block, 64, num_blocks[0], stride=1),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2))
        if rbfLast:
            self.rbf_lastconv = nn.Sequential(RBFConv2d(512,512,kernel_size = 1, stride = 1))
            self.dorbflast = True
        self.linear = nn.Sequential(nn.Linear(512*block.expansion, num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        if self.dorbflast:
            out = self.rbf_lastconv(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def get_features_before_rbf(self, x):
        out = self.features(x)
        out = F.avg_pool2d(out, 4)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, rbf = False, rbfLast = False, emb = False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if not rbf:
            self.conv1 = conv3x3(3,64)
            self.rbf = False
        else:
            self.rbf = True
            self.conv1 = RBFConv2d(3,64,kernel_size = 3, stride = 1, padding = 1)
            self.conv11 = RBFConv2d(64,64,kernel_size = 3, stride = 1, padding = 1)
        self.dorbflast = rbfLast 
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if rbfLast:
            self.rbf_lastconv = RBFConv2d(512,512,kernel_size = 1, stride = 1)
            self.dorbflast = True
        self.bn2 = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        if emb:
            self.do_emb = True
            self.fasttext = nn.Linear(512*block.expansion, 300)
        else:
            self.do_emb = False
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.rbf:
            out = self.conv11(self.conv1(x))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.dorbflast:
            out = self.rbf_lastconv(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if self.do_emb:
            emb1 = L2Norm()(self.fasttext(out))
        out = self.linear(out)
        if self.do_emb:
            return out,emb1
        return out
    def get_features_before_rbf(self, x):
        if self.rbf:
            out = self.conv11(self.conv1(x))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

def ResNet18RBF():
    return ResNet(PreActBlock, [2,2,2,2], rbf=True)

def ResNet18RBFLast():
    return ResNetRBFCLF(PreActBlock, [2,2,2,2], rbf=False, rbfLast=True)

def ResNet18():
    return ResNet(PreActBlock, [2,2,2,2])
def ResNet18_emb():
    return ResNet(PreActBlock, [2,2,2,2], emb = True)

def ResNet18Drop():
    return ResNetDrop(PreActBlock, [2,2,2,2])
def ResNet18DropOne():
    return ResNetDropOne(PreActBlock, [2,2,2,2])
def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
class DropoutOne(InplaceFunction):

    def __init__(self, p=0.5, train=False, inplace=False):
        super(DropoutOne, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.train = train
        self.inplace = inplace

    def _make_noise(self, input):
        return input.new().resize_as_(input)

    def forward(self, input):
        output = input.clone()

        if self.p > 0 and self.train:
            self.noise = self._make_noise(input)
            self.noise.bernoulli_(1 - self.p).div_(1 - self.p)
            if self.p == 1:
                self.noise.fill_(0)
            self.noise = self.noise.expand_as(input)
            self.addnoise = 1 - self.noise
            output.mul_(self.noise)
            output = output + self.addnoise
        return output

    def backward(self, grad_output):
        if self.p > 0 and self.train:
            return grad_output.mul(self.noise)
        else:
            return grad_output
class FeatureDropoutOne(DropoutOne):

    def _make_noise(self, input):
        return input.new().resize_(input.size(0), input.size(1),
                                   *repeat(1, input.dim() - 2))

class Dropout2dOne(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout2dOne, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return FeatureDropoutOne(self.p, self.training, self.inplace)(input)



class ResNetDropOne(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetDropOne, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.drop1 = Dropout2dOne(0.05)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.drop2 = Dropout2dOne(0.1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.drop3 = Dropout2dOne(0.1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.drop4 = Dropout2dOne(0.15)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.drop5 = Dropout2dOne(0.3)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.drop1(self.conv1(x))))
        out = self.drop2(self.layer1(out))
        out = self.drop3(self.layer2(out))
        out = self.drop4(self.layer3(out))
        out = self.layer4(out)
        out = self.drop5(F.avg_pool2d(out, 4))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetDrop(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetDrop, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.drop1 = nn.Dropout2d(0.05)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.drop2 = nn.Dropout2d(0.1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.drop3 = nn.Dropout2d(0.1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.drop4 = nn.Dropout2d(0.15)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.drop5 = nn.Dropout2d(0.3)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.drop1(self.conv1(x))))
        out = self.drop2(self.layer1(out))
        out = self.drop3(self.layer2(out))
        out = self.drop4(self.layer3(out))
        out = self.layer4(out)
        out = self.drop5(F.avg_pool2d(out, 4))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
