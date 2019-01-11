import torch
import torch.nn as nn

import torchvision

from torch.autograd import Variable

class styleEncoder(nn.Module):

    def __init__(self):
        super(styleEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=32, stride=1, padding=0),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=18, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        # B*1*64*64 -> B*8*33*33 -> B*16*16*16 -> B*32*8*8 -> B*64*4*4
        self.fc = nn.Linear(1024, 256)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x  = self.conv4(x)
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))

        x = self.fc(x)
        return x

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(256, 1024)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=18, stride=1, padding=0),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=32, stride=1, padding=0),
            nn.Sigmoid()
        )
        # B*128*4*4 -> B*32*8*8 -> B*16*16*16 -> B*8*33*33 -> B*1*64*64
    
    def forward(self, style, char):
        style = self.fc1(style)
        char = self.fc2(char)
        x = torch.cat((style, char), 1)
        
        x = x.view(x.size(0), 128, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x) * 255

        return x

class fontEncoder2(nn.Module):

    def __init__(self):
        super(fontEncoder2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=21, stride=2, padding=0),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=0),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU()
        )
        # B*1*64*64 -> B*16*22*22 -> B*32*8*8 -> B*64*4*4
        self.fc = nn.Linear(1024, 256)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))

        output = self.fc(x)
        return output

class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=18, stride=2, padding=0),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=8, stride=2, padding=0),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(972, 360),
            nn.ReLU(),
            nn.Linear(360, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))

        x = self.fc(x)
        return x


class fontEncoder(nn.Module):

    def __init__(self):
        super(fontEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=21, stride=2, padding=0),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=0),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU()
        )
        # B*1*64*64 -> B*16*22*22 -> B*32*8*8 -> B*64*4*4
        self.fc = nn.Linear(1024, 256)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))

        output = self.fc(x)
        return output


class classifier(nn.Module):

    def __init__(self):
        super(classifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 50), ############
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x