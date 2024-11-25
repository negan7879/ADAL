import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class GeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32):
        super(GeneratorA, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img


class GeneratorB_native(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """

    def __init__(self, nz=256, ngf=64, nc=3, img_size=64, slope=0.2):
        super(GeneratorB_native, self).__init__()
        if isinstance(img_size, (list, tuple)):
            self.init_size = (img_size[0] // 16, img_size[1] // 16)
        else:
            self.init_size = (img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf * 8 * self.init_size[0] * self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output

import numpy as np
class GeneratorB(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, nz=256, ngf=64, nc=3, img_size=64, slope=0.2, args = None):
        super(GeneratorB, self).__init__()
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:    
            self.init_size = ( img_size // 16, img_size // 16)
        self.args = args
        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Tanh(),
        )

        # self.main_2 = nn.Sequential(
        #
        #
        # )
        # le_name = "/work/zhangzherui/code/Data-Free-Adversarial-Distillation/label_emb/imagenet_le.pickle"
        le_name = "./label_emb/nyu_le.pickle"
        with open(le_name, "rb") as label_file:
            label_emb = pickle.load(label_file)
            label_emb = label_emb.cuda().float()
        self.label_emb = label_emb
        le_size = 512
        self.n1 = nn.BatchNorm1d(le_size)
        self.nl = 13
        self.nle = int(np.ceil(args.batch_size / self.nl )) # 13 is class
        le_emb_size = 1000
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for i in range(self.nle)])

        self.l1 = nn.Sequential(nn.Linear(le_size, ngf * 8 * self.init_size[0] * self.init_size[1] ))

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, targets, bias=None,  fuse = False):
        bs = targets.shape[0]
        num_it = targets.shape[0] // self.args.num_classes
        label_emb = self.label_emb
        if bias != None:
            # bias = self.n1(bias)
            label_emb = label_emb + 0.001 * bias

        le = label_emb[targets].detach()

        # le = self.label_emb[targets].detach()
        # le = self.sig1(le)
        nos = torch.randn(le.shape).cuda()
        le = nos + 0.01 * le
        # le = self.n1(le)
        # v = None
        # for i in range(self.nle):
        #     if (i + 1) * self.nl > le.shape[0]:
        #         sle = le[i * self.nl:]
        #     else:
        #         sle = le[i * self.nl:(i + 1) * self.nl]
        #     sv = self.le1[i](sle)
        #     if v is None:
        #         v = sv
        #     else:
        #         v = torch.cat((v, sv))

        out = self.l1(le)
        out = out.view(out.shape[0], -1, self.init_size[0], self.init_size[1])


        # proj = self.project(out)
        # proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        if fuse and torch.rand(1).item() < 0.2:
            fuse_before = self.main[:4]
            fuse_after = self.main[4:]
            fea_aug = fuse_before(out)
            for j in range(num_it):
                fea_tmp = fea_aug[j * self.args.num_classes:(j + 1) * self.args.num_classes]
                indices = torch.randperm(self.args.num_classes)

                fea_aug[j * self.args.num_classes:(j + 1) * self.args.num_classes] = self.channel_mix_batch(fea_tmp[indices])
            output = fuse_after(fea_aug)
            # print("hello")

        else:
            output = self.main(out)
        # if fuse == True:
        #     output = self.channel_mix_batch(output)
        # output = self.main_2(output)



        return output

    def channel_mix_feature(self, a, b):

        nc = a.size()[0]
        picks = np.random.choice(nc, nc // 2, replace=False)
        se2 = set(range(nc)) - set(picks)
        unpicks = list(se2)
        cmask1 = torch.zeros([nc], device=next(self.parameters()).device).scatter_(0, torch.LongTensor(picks).cuda(), 1)
        cmask2 = torch.zeros([nc], device=next(self.parameters()).device).scatter_(0, torch.LongTensor(unpicks).cuda(), 1)
        viewidxs = [nc] + [1 for i in range(len(list(a.size())) - 1)]
        aug_feature = a * cmask1.view(*viewidxs) + b * cmask2.view(*viewidxs)

        return aug_feature

    def channel_mix_batch(self, feature):

        nfeature = feature.size()[0]

        ansfeature = feature.clone().detach()
        # anslabels = labels.clone().detach()

        for i in range(nfeature):
            j = nfeature - 1 - i

            ansfeature[i] = self.channel_mix_feature(feature[i], feature[j])

            # anslabels[i] = (labels[i] + labels[j]) / 2

        return ansfeature

