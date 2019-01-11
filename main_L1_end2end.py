import torch
import torch.nn as nn

import numpy as np
import random

from networks import styleEncoder, Decoder, discriminator, fontEncoder, classifier
from utils import get_style_set, randomPickImage


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

s_enc = styleEncoder().cuda()
dec = Decoder().cuda()
dis = discriminator().cuda()
enc = fontEncoder().cuda()
cla = classifier().cuda()

s_MSE = nn.MSELoss().cuda()
adv_loss = nn.BCELoss().cuda()
cla_loss = nn.CrossEntropyLoss().cuda()
gen_loss = nn.L1Loss().cuda()

D_optimizer = torch.optim.Adam(dis.parameters(), lr=learning_rate)
G_optimizer = torch.optim.Adam(list(s_enc.parameters()) + list(dec.parameters()) + list(enc.parameters()) + list(cla.parameters()), lr= learning_rate)


for epoch in range(NUM_EPOCH):

    print("current epoch: ", epoch)

    for index, image in enumerate(style_set):
        valid = torch.Tensor(np.array([[1]])).cuda()
        fake = torch.Tensor(np.array([[0]])).cuda()

        u_index = random.choice(range(len(readList)))
        pic = randomPickImage(base + readList[u_index]).cuda()

        image = image[0].cuda()
        u_label = torch.Tensor([u_index]).long().cuda()

        # train G #
        G_optimizer.zero_grad()

        u = enc(pic)
        s = s_enc(image)
        fake_image = dec(s, u) # reconstruct
        s_prime = s_enc(fake_image) # twice forward
        validate = dis(fake_image) # fool the dis
        cla_output = cla(u) # classification

        G_loss = s_MSE(s_prime, s) + adv_loss(validate, valid) + gen_loss(fake_image, pic) + cla_loss(cla_output, u_label)
        G_loss.backward(retain_graph=True)
        G_optimizer.step()

        # train D #
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
    torch.save(enc, "f_enc.pkl")
    torch.save(cla, "cla.pkl")









