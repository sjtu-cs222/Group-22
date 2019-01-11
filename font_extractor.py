import torch
import torch.nn as nn

import torchvision

import numpy as np

from torch.autograd import Variable
from utils import getBatch

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



base = "data/"
readList = open("remaining.txt").read().split()[:50] ##########
batch_size = 64
NUM_EPOCH = 80 # 1200
learning_rate = 0.0003

enc = fontEncoder().cuda()
cla = classifier().cuda()

loss_function = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(list(enc.parameters()) + list(cla.parameters()), lr=learning_rate)

train_set = getBatch(base, readList, BATCH_SIZE=batch_size)

for epoch in range(NUM_EPOCH):
    
    print("current epoch: ", epoch)

    for index, (image, label) in enumerate(train_set):
        optimizer.zero_grad()

        image = image.cuda()
        label = label.cuda()

        embedding = enc(image)
        output = cla(embedding)

        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()

        if index%10 == 0:
            print("case ", index, ", current loss = %0.5f" % loss.item())
    torch.save(enc, "enc1.pkl")
    torch.save(cla, "cla1.pkl")
'''
base = "gen_image/"

START = 108

END = 129 # 3982
NUM_EPOCH = 1200
learning_rate = 0.0003

enc = fontEncoder().cuda()
cla = classifier().cuda()

loss_function = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(list(enc.parameters()) + list(cla.parameters()), lr=learning_rate)

# loss_history = [0]*3874
# max_loss = [0]*3874
print("reading data.")
data = []
labels = []
for i in range(START, END):

    pics = getData(base, i)
    label = np.zeros(pics.shape[0], dtype=np.int32)
    label[:] = i - START

    data.append(pics)
    labels.append(label)

for epoch in range(NUM_EPOCH):
    
    print("current epoch: ", epoch)

    for i in range(START, START + 20):

        optimizer.zero_grad()

        pic = data[i - START]
        label = labels[i - START]

        pic = Variable(torch.Tensor(pic)).cuda()
        label = Variable(torch.Tensor(label)).cuda().long()

        embedding = enc(pic)
        output = cla(embedding)

        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()

        if i%5 == 0:
            print("case ", i, ", current loss = %0.5f" % loss.item())
    torch.save(enc, "enc1.pkl")
    torch.save(cla, "cla1.pkl")
'''