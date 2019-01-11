import os
import numpy as np
from PIL import Image
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def getBatch(base, readList, BATCH_SIZE=128):

    images = []
    labels = []
    count = 0

    for i in readList:
        tmpPath = base + i
        for im in os.listdir(tmpPath):
            imPath = tmpPath + "/" + im
            try:
                pic = np.array(Image.open(imPath))
                labels.append(count)
                images.append([pic])
            except:
                continue
        count += 1
    
    images = torch.Tensor(np.array(images))
    labels = torch.Tensor(np.array(labels)).long()
    train = TensorDataset(images, labels)
    train_set = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
    return train_set

def getData(base, readList):
    images = []

    for i in readList:
        tmpPath = base + i
        current = []

        for im in os.listdir(tmpPath):
            imPath = tmpPath + "/" + im
            try:
                pic = np.array(Image.open(imPath))
                current.append([pic])
            except:
                continue
        images.append(np.array(current))
    return images

def MeanSquareError(a, b):
    return np.mean(np.abs(a - b))


def generatePositive(base, readList):
    a = []
    b = []
    count = 0

    for i in readList:
        images = []
        tmpPath = base + i
        for im in os.listdir(tmpPath):
            imPath = tmpPath + "/" + im
            try:
                pic = np.array(Image.open(imPath))
                images.append(pic)
            except:
                continue
        for i in range(len(images) - 1):
            for j in images[i+1::5]:
                a.append([images[i]])
                b.append([j])
                count += 1
    a = torch.Tensor(np.array(a))
    b = torch.Tensor(np.array(b))
    return a, b, count
    
def generateNegative(base, readList, num_samples):
    images = {}

    for i in readList:
        tmpPath = base + i
        for im in os.listdir(tmpPath):
            if im not in images.keys():
                images[im] = []
            imPath = tmpPath + "/" + im
            pic = np.array(Image.open(imPath))
            images[im].append(pic)
    
    a = []
    b = []
    domains = list(images.keys())
    sampelPerDomain = num_samples//len(domains)

    for i in domains:
        if i != domains[-1]:
            n = sampelPerDomain
        else:
            n = num_samples - sampelPerDomain * (len(domains) - 1)

        length = len(images[i])
        while n > 0:
            index1 = random.choice(range(length))
            index2 = random.choice(range(length))
            while index1 == index2:
                index2 = random.choice(range(length))

            a.append([images[i][index1]])
            b.append([images[i][index2]])
            n -= 1

    a = torch.Tensor(np.array(a))
    b = torch.Tensor(np.array(b))
    return a, b

def create_dataset(base, readList, BATCH_SIZE=60):
    print("creating positive set.")
    pos_a, pos_b, num = generatePositive(base, readList)
    print("creating negative set.")
    neg_a, neg_b = generateNegative(base, readList, num)
    print(pos_a.shape)
    print(pos_b.shape)
    print(neg_a.shape)
    print(neg_b.shape)

    # train_pos = TensorDataset(pos_a, pos_b)
    # train_neg = TensorDataset(neg_a, neg_b)

    # pos_set = DataLoader(dataset=train_pos, batch_size=BATCH_SIZE, shuffle=True)
    # neg_set = DataLoader(dataset=train_neg, batch_size=BATCH_SIZE, shuffle=True)
    train = TensorDataset(pos_a, pos_b, neg_a, neg_b)
    train_set = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
    return train_set # pos_set, neg_set


def get_style_set(path):
    images = []
    for im in os.listdir(path):
        imPath = path + im
        pic = np.array(Image.open(imPath))
        images.append([pic])
    images = torch.Tensor(images)
    style_samples = TensorDataset(images)
    style_set = DataLoader(dataset=style_samples, shuffle=True)
    return style_set


def randomPickImage(path):
    im = random.choice(os.listdir(path))
    imPath = path + "/" + im
    pic = [[np.array(Image.open(imPath))]]
    return torch.Tensor(pic)
    