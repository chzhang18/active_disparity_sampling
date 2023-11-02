import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])

        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class AdjointNet(nn.Module):
    def __init__(self, ngf=64, input_nc=3, output_nc=1):
        super(AdjointNet, self).__init__()
        #initialize layers
        self.adjoint_convlayer1 = unet_conv(input_nc, ngf)
        self.adjoint_convlayer2 = unet_conv(ngf, ngf * 2)
        self.adjoint_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.adjoint_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.adjoint_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.adjoint_upconvlayer1 = unet_upconv(512, ngf * 8)
        self.adjoint_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.adjoint_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.adjoint_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.adjoint_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True)
        #self.conv1x1 = create_conv(512, 8, 1, 0) #reduce dimension of extracted visual features

    def forward(self, x):
        adjoint_conv1feature = self.adjoint_convlayer1(x)
        adjoint_conv2feature = self.adjoint_convlayer2(adjoint_conv1feature)
        adjoint_conv3feature = self.adjoint_convlayer3(adjoint_conv2feature)
        adjoint_conv4feature = self.adjoint_convlayer4(adjoint_conv3feature)
        adjoint_conv5feature = self.adjoint_convlayer5(adjoint_conv4feature)
        
        adjoint_upconv1feature = self.adjoint_upconvlayer1(adjoint_conv5feature)
        adjoint_upconv2feature = self.adjoint_upconvlayer2(torch.cat((adjoint_upconv1feature, adjoint_conv4feature), dim=1))
        adjoint_upconv3feature = self.adjoint_upconvlayer3(torch.cat((adjoint_upconv2feature, adjoint_conv3feature), dim=1))
        adjoint_upconv4feature = self.adjoint_upconvlayer4(torch.cat((adjoint_upconv3feature, adjoint_conv2feature), dim=1))
        depth_prediction = self.adjoint_upconvlayer5(torch.cat((adjoint_upconv4feature, adjoint_conv1feature), dim=1))
        return depth_prediction

