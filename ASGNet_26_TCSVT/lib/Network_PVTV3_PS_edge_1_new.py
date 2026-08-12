import timm
from lib.pvt_v2 import pvt_v2_b3
import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.module3_NEW_PS import SNP_module
from lib.module3_NEW_PS_edge import MSE, DCI_decoder_1,DCI_decoder_2,DCI_decoder_3,DCI_decoder_4

'''
backbone: resnet50
'''


class Network(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=128):
        super(Network, self).__init__()
        #self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

        self.shared_encoder = pvt_v2_b3()

        pretrained_dict = torch.load('/pvt_v2_b3.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.shared_encoder.state_dict()}
        self.shared_encoder.load_state_dict(pretrained_dict)

        self.dePixelShuffle = torch.nn.PixelShuffle(2)

        self.up = nn.Sequential(
            nn.Conv2d(channels//4, channels, kernel_size=1),nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),nn.BatchNorm2d(channels),nn.ReLU(True)
        )

        self.SNP_5 = SNP_module(512, channels)
        self.SNP_4 = SNP_module(320 + channels, channels)
        self.SNP_3 = SNP_module(128 + channels, channels)
        self.SNP_2 = SNP_module(64 + channels, channels)

        self.MSE = MSE(512, channels)

        self.DCI_1 = DCI_decoder_1(channels)
        self.DCI_2 = DCI_decoder_2(channels)
        self.DCI_3 = DCI_decoder_3(channels)
        self.DCI_4 = DCI_decoder_4(channels)


    def forward(self, x):
        image = x

        en_feats = self.shared_encoder(x)
        x4, x3, x2, x1 = en_feats
        x4_1 = x4
        #p1 = self.Global6(x4)
        #x5_4 = p1
        #x5_4_1 = x5_4.expand(-1, self.channel, -1, -1)

        x4   = self.SNP_5(x4)
        x4_2 = x4
        x4_up = self.up(self.dePixelShuffle(x4))

        p1 = self.MSE(x4_1)
        x5_4 = p1


        #x4_3 = self.up(x4)
        x3   = self.SNP_4(torch.cat((x3,x4_up),1))
        x3_up = self.up(self.dePixelShuffle(x3))

        #x3_2 = self.up(x3)
        x2   = self.SNP_3(torch.cat((x2,x3_up),1))
        x2_up = self.up(self.dePixelShuffle(x2))


        #x2_1 = self.up(x2)
        x1   = self.SNP_2(torch.cat((x1,x2_up),1))


        x4,e4 = self.DCI_1(x4,x5_4)
        x3,e3 = self.DCI_2(x3,x4,x5_4)
        x2,e2 = self.DCI_3(x2,x3,x4,x5_4)
        x1,e1 = self.DCI_4(x1,x2,x3,x4,x5_4)


        p0 = F.interpolate(p1, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(x1, size=image.size()[2:], mode='bilinear', align_corners=True)
        e4 = F.interpolate(e4, size=image.size()[2:], mode='bilinear', align_corners=True)
        e3 = F.interpolate(e3, size=image.size()[2:], mode='bilinear', align_corners=True)
        e2 = F.interpolate(e2, size=image.size()[2:], mode='bilinear', align_corners=True)
        e1 = F.interpolate(e1, size=image.size()[2:], mode='bilinear', align_corners=True)


        return p0, f4, f3, f2, f1, e2, e3, e4, e1

if __name__ == '__main__':
    image = torch.rand(2, 3, 384, 384).cuda()
    model = Network(96).cuda()
    pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = model(image)
    print(pred_0.shape)
    print(f4.shape)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)
    print(bound_f4.shape)
    print(bound_f3.shape)
    print(bound_f2.shape)
    print(bound_f1.shape)
