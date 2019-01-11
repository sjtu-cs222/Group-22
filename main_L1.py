import torch
import torch.nn as nn

import numpy as np
import random

from torch.autograd import Variable

from networks import styleEncoder, Decoder, fontEncoder2, discriminator, fontEncoder, classifier
from utils import get_style_set, randomPickImage

s_enc = styleEncoder().cuda()
dec = Decoder().cuda()
dis = discriminator().cuda()

enc = torch.load("enc1.pkl")

##########################
# base image             #
base = "data/"
readList = open("remaining.txt").read().split()[:50]
##########################

##########################
# get hand written data  #
print("generating stylization set.")
style_base = "stylization_set/another/"
style_set = get_style_set(style_base)
##########################

print("initializing model.")
NUM_EPOCH = 200 ################
learning_rate = 0.0003

s_MSE = nn.MSELoss().cuda()
adv_loss = nn.BCELoss().cuda()
gen_loss = nn.L1Loss().cuda()

D_optimizer = torch.optim.Adam(dis.parameters(), lr=learning_rate)
G_optimizer = torch.optim.Adam(list(s_enc.parameters()) + list(dec.parameters()), lr= learning_rate)


for epoch in range(NUM_EPOCH):

    print("current epoch: ", epoch)

    for index, image in enumerate(style_set):
        valid = torch.Tensor(np.array([[1]])).cuda()
        fake = torch.Tensor(np.array([[0]])).cuda()

        u_index = random.choice(range(len(readList)))
        pic = randomPickImage(base + readList[u_index]).cuda()

        image = image[0].cuda()
        u = enc(pic)
        u_label = torch.Tensor([u_index]).long().cuda()

        # train G #
        G_optimizer.zero_grad()

        s = s_enc(image)
        fake_image = dec(s, u)
        s_prime = s_enc(fake_image)
        validate = dis(fake_image)

        G_loss = s_MSE(s_prime, s) + adv_loss(validate, valid) + gen_loss(fake_image, pic)
        G_loss.backward(retain_graph=True)
        G_optimizer.step()

        # train_D #
        D_optimizer.zero_grad()

        d_real = dis(image)
        d_fake = dis(fake_image)

        D_loss = adv_loss(d_real, valid) + adv_loss(d_fake, fake)
        D_loss.backward()
        D_optimizer.step()

        if index%30 == 0:
            print("case ", index, ", current D_loss = %0.5f" % D_loss.item(), ", current G_loss = %0.5f" % G_loss.item())
    torch.save(s_enc, "s_enc.pkl")
    torch.save(dec, "dec.pkl")
    torch.save(dis, "dis.pkl")