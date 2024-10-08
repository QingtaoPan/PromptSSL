# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Vit import VisionTransformer, Reconstruct
from .pixlevel import PixLevelModule


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):  # 1024, 256, 2
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)  # 1024/2 = 512
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)  # [b, 512, 28,28]
        skip_x_att = self.pixModule(skip_x)  # [b, 512, 28,28]
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension  # [b, 1024, 28,28]
        return self.nConvs(x)  # [b, 256, 28,28]


class LViT(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels  # 3
        self.n_classes = n_classes  # 1
        in_channels = config.base_channel  # 64
        self.inc = ConvBatchNorm(n_channels, in_channels)  # (3, 64)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)  # 1024, 256
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)  # 512, 128
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)  # 256, 64
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)  # 128, 64
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.pix_module1 = PixLevelModule(64)
        self.pix_module2 = PixLevelModule(128)
        self.pix_module3 = PixLevelModule(256)
        self.pix_module4 = PixLevelModule(512)
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

    def forward(self, x, text):
        device = x.device
        x = x.float()  # x [4,3,224,224]
        x1 = self.inc(x)  # x1 [4, 64, 224, 224]
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2)   # [4, num_words, 512]
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)  # [4, num_words, 256]
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)  # [4, num_words, 128]
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)  # [4, num_words, 64]
        y1 = self.downVit(x1, x1, text1)  # [b, 14*14, 64]====([b, 64, 224, 224], [b, 64, 224, 224], [b, num_words, 64])
        x2 = self.down1(x1)  # [b, 128, 112, 112]
        y2 = self.downVit1(x2, y1, text2)  # [b, 14*14, 128]====([b, 128, 112, 112], [b, 14*14, 64])
        x3 = self.down2(x2)  # [b, 256, 56, 56]
        y3 = self.downVit2(x3, y2, text3)  # [b, 14*14, 256]====([b, 256, 56, 56], [b, 14*14, 128])
        x4 = self.down3(x3)  # [b, 512, 28, 28]
        y4 = self.downVit3(x4, y3, text4)  # [b, 14*14, 512]====([b, 512, 28, 28], [b, 14*14, 256])
        x5 = self.down4(x4)  # [b, 512, 14, 14]
        y4 = self.upVit3(y4, y4, text4, True)  # [b, 14*14, 512]====([b, 14*14, 512], [b, 14*14, 512])
        y3 = self.upVit2(y3, y4, text3, True)  # [b, 14*14, 256]====([b, 14*14, 256], [b, 14*14, 512])
        y2 = self.upVit1(y2, y3, text2, True)  # [b, 14*14, 128]====([b, 14*14, 128], [b, 14*14, 256])
        y1 = self.upVit(y1, y2, text1, True)  # [b, 14*14, 64]====([b, 14*14, 64], [b, 14*14, 128])
        x1 = self.reconstruct1(y1) + x1  # [b, 64, 224, 224]
        x2 = self.reconstruct2(y2) + x2  # [b, 128, 112, 112]
        x3 = self.reconstruct3(y3) + x3  # [b, 256, 56, 56]
        x4 = self.reconstruct4(y4) + x4  # [b, 512, 28, 28]
        x = self.up4(x5, x4)  # [b, 256, 28, 28]==([b, 512, 14, 14], [b, 512, 28, 28])
        x = self.up3(x, x3)  # [b, 128, 56, 56]==([b, 256, 28, 28], [b, 256, 56, 56])
        x = self.up2(x, x2)  # [b, 64, 112, 112]==([b, 128, 56, 56], [b, 128, 112, 112])
        x = self.up1(x, x1)  # [b, 64, 224, 224]==([b, 64, 112, 112], [b, 64, 224, 224])
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))  # [b, 1, 224, 224]-->[b, 1, 224, 224]
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
        return logits
