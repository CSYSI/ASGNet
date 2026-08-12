import torch
import torch.nn as nn
import torch.nn.functional as F


class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.sa2 = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 5, padding=2, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.sa(x)
        out2 = self.sa2(x)
        y = x * out1 + x * out2
        return y

class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)+self.mp(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class AM(nn.Module):
    def __init__(self, channels):
        super(AM, self).__init__()
        self.CA = CA()
        self.SA = SA(channels)

    def forward(self, x):
        x_res = x
        x = self.CA(x)
        x = self.SA(x)

        return x+x_res


class DASPP_F(nn.Module):
    def __init__(self, inchannels, outchannels=128):

        super(DASPP_F, self).__init__()


        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1),nn.BatchNorm2d(outchannels),nn.ReLU(True)
        )
        self.branch5 = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1), nn.BatchNorm2d(outchannels),nn.ReLU(True))
        self.branch0 = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1),nn.BatchNorm2d(outchannels),nn.ReLU(True))
        self.branch1 = nn.Sequential(nn.Conv2d(inchannels+outchannels, outchannels, kernel_size=3, stride=1, padding=3,dilation=3), nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch2 = nn.Sequential(nn.Conv2d(inchannels+outchannels*2, outchannels, kernel_size=3, stride=1, padding=6,dilation=6),nn.BatchNorm2d(outchannels),nn.ReLU(True))
        self.branch3 = nn.Sequential(nn.Conv2d(inchannels+outchannels*3, outchannels, kernel_size=3, stride=1, padding=12,dilation=12),nn.BatchNorm2d(outchannels),nn.ReLU(True))
        self.branch4 = nn.Sequential(nn.Conv2d(inchannels+outchannels*4, outchannels, kernel_size=3, stride=1, padding=24, dilation=24),nn.BatchNorm2d(outchannels),nn.ReLU(True))

        self.out = nn.Sequential(
            nn.Conv2d(outchannels * 7, outchannels, kernel_size=1), nn.BatchNorm2d(outchannels), nn.ReLU(True),
            nn.Conv2d(outchannels, outchannels//2, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, 1)
        )
        self.norm = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(outchannels)
        self.joint_attention = AM(outchannels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        branch0 = self.branch0(x)
        branch1 = self.branch1(torch.cat((x,branch0),1))
        branch2 = self.branch2(torch.cat((x,branch0,branch1),1))
        branch3 = self.branch3(torch.cat((x,branch0,branch1,branch2),1))
        branch4 = self.branch4(torch.cat((x,branch0,branch1,branch2,branch3),1))
        branch5_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(self.branch5(x).float()).real)))))
        out = torch.cat([branch_main, branch0, branch1, branch2, branch3, branch4, branch5_f], 1)
        out = self.out(out)
        return out


class MSE(nn.Module):
    def __init__(self, inchannels, outchannels):

        super(MSE, self).__init__()

        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1), nn.BatchNorm2d(outchannels), nn.ReLU(True)
        )
        self.branch0_f = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch0 = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1), nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch1 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=9, dilation=9),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch4 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch5 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=15, dilation=15),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch6 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=18, dilation=18),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))

        self.out = nn.Sequential(
            nn.Conv2d(outchannels * 8, outchannels, kernel_size=1), nn.BatchNorm2d(outchannels), nn.ReLU(True),
        )

        self.out_1 = nn.Sequential(
            nn.Conv2d(outchannels, 1, kernel_size=1),
        )

        self.norm = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(outchannels)
        self.joint_attention = AM(outchannels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        x_reduce = self.branch0(x)
        branch1 = self.branch1(x_reduce)
        branch2 = self.branch2(x_reduce+branch1)
        branch3 = self.branch3(x_reduce+branch1+branch2)
        branch4 = self.branch4(x_reduce+branch1+branch2+branch3)
        branch5 = self.branch5(x_reduce+branch1+branch2+branch3+branch4)
        branch6 = self.branch6(x_reduce+branch1+branch2+branch3+branch4+branch5)

        branch5_f = self.relu(
            self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(self.branch0_f(x).float()).real)))))
        out = torch.cat([branch_main, branch1, branch2, branch3, branch4, branch5_f, branch5, branch6], 1)
        out = self.out_1(self.out(out)+x_reduce)
        return out


class MSE_noafs(nn.Module):
    def __init__(self, inchannels, outchannels):

        super(MSE_noafs, self).__init__()

        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1), nn.BatchNorm2d(outchannels), nn.ReLU(True)
        )
        self.branch0_f = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch0 = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1), nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch1 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=9, dilation=9),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch4 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch5 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=15, dilation=15),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch6 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=18, dilation=18),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))

        self.out = nn.Sequential(
            nn.Conv2d(outchannels * 7, outchannels, kernel_size=1), nn.BatchNorm2d(outchannels), nn.ReLU(True),
        )

        self.out_1 = nn.Sequential(
            nn.Conv2d(outchannels, 1, kernel_size=1),
        )

        #self.norm = nn.BatchNorm2d(outchannels)
        #self.relu = nn.ReLU(outchannels)
        #self.joint_attention = AM(outchannels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        x_reduce = self.branch0(x)
        branch1 = self.branch1(x_reduce)
        branch2 = self.branch2(x_reduce+branch1)
        branch3 = self.branch3(x_reduce+branch1+branch2)
        branch4 = self.branch4(x_reduce+branch1+branch2+branch3)
        branch5 = self.branch5(x_reduce+branch1+branch2+branch3+branch4)
        branch6 = self.branch6(x_reduce+branch1+branch2+branch3+branch4+branch5)

        #branch5_f = self.relu(
            #self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(self.branch0_f(x).float()).real)))))
        out = torch.cat([branch_main, branch1, branch2, branch3, branch4, branch5, branch6], 1)
        out = self.out_1(self.out(out)+x_reduce)
        return out






class MSE_1(nn.Module):
    def __init__(self, inchannels, outchannels):

        super(MSE_1, self).__init__()

        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1), nn.BatchNorm2d(outchannels), nn.ReLU(True)
        )
        self.branch0_f = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch0 = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1), nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch1 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch4 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch5 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch6 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))

        self.out = nn.Sequential(
            nn.Conv2d(outchannels * 8, outchannels, kernel_size=1), nn.BatchNorm2d(outchannels), nn.ReLU(True),
        )

        self.out_1 = nn.Sequential(
            nn.Conv2d(outchannels, 1, kernel_size=1),
        )

        self.norm = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(outchannels)
        self.joint_attention = AM(outchannels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        x_reduce = self.branch0(x)
        branch1 = self.branch1(x_reduce)
        branch2 = self.branch2(x_reduce+branch1)
        branch3 = self.branch3(x_reduce+branch1+branch2)
        branch4 = self.branch4(x_reduce+branch1+branch2+branch3)
        branch5 = self.branch5(x_reduce+branch1+branch2+branch3+branch4)
        branch6 = self.branch6(x_reduce+branch1+branch2+branch3+branch4+branch5)

        branch5_f = self.relu(
            self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(self.branch0_f(x).float()).real)))))
        out = torch.cat([branch_main, branch1, branch2, branch3, branch4, branch5_f, branch5, branch6], 1)
        out = self.out_1(self.out(out)+x_reduce)
        return out


class MSE_2(nn.Module):
    def __init__(self, inchannels, outchannels):

        super(MSE_2, self).__init__()

        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1), nn.BatchNorm2d(outchannels), nn.ReLU(True)
        )
        self.branch0_f = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch0 = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1), nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch1 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch4 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch5 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch6 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))

        self.out = nn.Sequential(
            nn.Conv2d(outchannels * 8, outchannels, kernel_size=1), nn.BatchNorm2d(outchannels), nn.ReLU(True),
        )

        self.out_1 = nn.Sequential(
            nn.Conv2d(outchannels, 1, kernel_size=1),
        )

        self.norm = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(outchannels)
        self.joint_attention = AM(outchannels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        x_reduce = self.branch0(x)
        branch1 = self.branch1(x_reduce)
        branch2 = self.branch2(x_reduce+branch1)
        branch3 = self.branch3(x_reduce+branch1+branch2)
        branch4 = self.branch4(x_reduce+branch1+branch2+branch3)
        branch5 = self.branch5(x_reduce+branch1+branch2+branch3+branch4)
        branch6 = self.branch6(x_reduce+branch1+branch2+branch3+branch4+branch5)

        branch5_f = self.relu(
            self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(self.branch0_f(x).float()).real)))))
        out = torch.cat([branch_main, branch1, branch2, branch3, branch4, branch5_f, branch5, branch6], 1)
        out = self.out_1(self.out(out)+x_reduce)
        return out


class MSE_3(nn.Module):
    def __init__(self, inchannels, outchannels):

        super(MSE_3, self).__init__()

        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1), nn.BatchNorm2d(outchannels), nn.ReLU(True)
        )
        self.branch0_f = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch0 = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1), nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch1 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch4 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch5 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))
        self.branch6 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=14, dilation=14),
            nn.BatchNorm2d(outchannels), nn.ReLU(True))

        self.out = nn.Sequential(
            nn.Conv2d(outchannels * 8, outchannels, kernel_size=1), nn.BatchNorm2d(outchannels), nn.ReLU(True),
        )

        self.out_1 = nn.Sequential(
            nn.Conv2d(outchannels, 1, kernel_size=1),
        )

        self.norm = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(outchannels)
        self.joint_attention = AM(outchannels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        x_reduce = self.branch0(x)
        branch1 = self.branch1(x_reduce)
        branch2 = self.branch2(x_reduce+branch1)
        branch3 = self.branch3(x_reduce+branch1+branch2)
        branch4 = self.branch4(x_reduce+branch1+branch2+branch3)
        branch5 = self.branch5(x_reduce+branch1+branch2+branch3+branch4)
        branch6 = self.branch6(x_reduce+branch1+branch2+branch3+branch4+branch5)

        branch5_f = self.relu(
            self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(self.branch0_f(x).float()).real)))))
        out = torch.cat([branch_main, branch1, branch2, branch3, branch4, branch5_f, branch5, branch6], 1)
        out = self.out_1(self.out(out)+x_reduce)
        return out














class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sa(x)
        y = x * out
        return y

class Edge_EH(nn.Module):
    def __init__(self, channels):
        super(Edge_EH, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1), nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, groups=channels), nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1), nn.BatchNorm2d(channels), nn.ReLU(True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1), nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, stride=1, groups=channels), nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1), nn.BatchNorm2d(channels), nn.ReLU(True))

        self.SA = SA(channels)

        self.out = nn.Sequential(
            nn.Conv2d(channels*2,channels,kernel_size=1), nn.BatchNorm2d(channels), nn.ReLU(True)
        )

    def forward(self, x):
        x1 = self.SA(self.conv3(x))
        x2 = self.SA(self.conv5(x))
        y  = self.out(torch.cat((x1,x2),1))
        return y

class Module_3_1(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Module_3_1, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1), nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1), nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=5, padding=2, stride=1),nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels//2, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels//2),nn.ReLU(True),
            nn.Conv2d(mid_channels//2, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels*2, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels),nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)
    def forward(self, X, prior_cam):

        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)  # 2,1,12,12->2,1,48,48
        FI  = X

        yt  = self.reduce(torch.cat((FI,prior_cam.expand(-1, X.size()[1], -1, -1)),1))

        out_edge = self.edge(yt)
        edge    = self.out_edge(out_edge)
        edge    = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce(torch.cat((yt_t,yt_f),1))


        r_prior_cam= -1 * (torch.sigmoid(prior_cam)) + 1
        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([out_edge, y_r, yt_out], dim=1)  # 2,128,48,48

        y = self.out(cat2)
        y = y + prior_cam
        return y,edge


    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out


class DCI_decoder_1(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_1, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1, groups=in_channels), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear', align_corners=True)  # 2,1,12,12->2,1,48,48
        FI = X

        yt = self.reduce(torch.cat((FI, prior_cam.expand(-1, X.size()[1], -1, -1)), 1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt) + self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce(torch.cat((yt_t, yt_f), 1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1
        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([out_edge, y_r, yt_out], dim=1)  # 2,128,48,48

        y = self.out(cat2)
        y = y + prior_cam
        return y, edge








    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class DCI_decoder_1_noafs(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_1_noafs, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1, groups=in_channels), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear', align_corners=True)  # 2,1,12,12->2,1,48,48
        FI = X

        yt = self.reduce(torch.cat((FI, prior_cam.expand(-1, X.size()[1], -1, -1)), 1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt) + self.conv5(yt)
        #yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        #yt_out = self.reduce(torch.cat((yt_t, yt_f), 1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1
        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([out_edge, y_r, yt_t], dim=1)  # 2,128,48,48

        y = self.out(cat2)
        y = y + prior_cam
        return y, edge








    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out
class DCI_decoder_1_noedge(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_1_noedge, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',
                                  align_corners=True)  # 2,1,12,12->2,1,48,48
        FI = X

        yt = self.reduce(torch.cat((FI, prior_cam.expand(-1, X.size()[1], -1, -1)), 1))

        #out_edge = self.edge(yt)
        #edge = self.out_edge(out_edge)
        #edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt) + self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce(torch.cat((yt_t, yt_f), 1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1
        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out], dim=1)  # 2,128,48,48

        y = self.out(cat2)
        y = y + prior_cam
        return y

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class Module_3_2(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Module_3_2, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1), nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, stride=1),nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1), nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels // 2, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt  = self.reduce(torch.cat((FI,prior_cam.expand(-1, X.size()[1], -1, -1),x1_prior_cam.expand(-1, X.size()[1], -1, -1)),1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce1(torch.cat((yt_t,yt_f),1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1

        r1_prior_cam = -1 * (torch.sigmoid(x1_prior_cam)) + 1

        r_prior_cam = r_prior_cam + r1_prior_cam

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out,out_edge], dim=1)  #

        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam
        return y,edge
    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class DCI_decoder_2(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_2, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt  = self.reduce(torch.cat((FI,prior_cam.expand(-1, X.size()[1], -1, -1),x1_prior_cam.expand(-1, X.size()[1], -1, -1)),1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce1(torch.cat((yt_t,yt_f),1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1

        r1_prior_cam = -1 * (torch.sigmoid(x1_prior_cam)) + 1

        r_prior_cam = r_prior_cam + r1_prior_cam

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out,out_edge], dim=1)  #

        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam
        return y,edge
    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class DCI_decoder_2_noafs(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_2_noafs, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt  = self.reduce(torch.cat((FI,prior_cam.expand(-1, X.size()[1], -1, -1),x1_prior_cam.expand(-1, X.size()[1], -1, -1)),1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt) + self.conv5(yt)
        #yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        #yt_out = self.reduce1(torch.cat((yt_t,yt_f),1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1

        r1_prior_cam = -1 * (torch.sigmoid(x1_prior_cam)) + 1

        r_prior_cam = r_prior_cam + r1_prior_cam

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_t, out_edge], dim=1)  #

        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam
        return y,edge
    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out
class DCI_decoder_2_noedge(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_2_noedge, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt  = self.reduce(torch.cat((FI,prior_cam.expand(-1, X.size()[1], -1, -1),x1_prior_cam.expand(-1, X.size()[1], -1, -1)),1))

        #out_edge = self.edge(yt)
        #edge = self.out_edge(out_edge)
        #edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce1(torch.cat((yt_t,yt_f),1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1

        r1_prior_cam = -1 * (torch.sigmoid(x1_prior_cam)) + 1

        r_prior_cam = r_prior_cam + r1_prior_cam

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out], dim=1)  #

        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam
        return y
    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class Module_3_3(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Module_3_3, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1), nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1), nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels // 2, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 4, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, x2, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        x2_prior_cam = F.interpolate(x2, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt = self.reduce(torch.cat((FI,prior_cam.expand(-1, X.size()[1], -1, -1),x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)),1))
        #yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)],dim=1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce1(torch.cat((yt_t,yt_f),1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1


        r1_prior_cam1 = -1 * (torch.sigmoid(x1_prior_cam)) + 1


        r2_prior_cam2 = -1 * (torch.sigmoid(x2_prior_cam)) + 1


        r_prior_cam = r_prior_cam + r1_prior_cam1+r2_prior_cam2

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out,out_edge], dim=1)
        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam + x2_prior_cam
        return y,edge
    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class DCI_decoder_3(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_3, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, x2, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        x2_prior_cam = F.interpolate(x2, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt = self.reduce(torch.cat((FI,prior_cam.expand(-1, X.size()[1], -1, -1),x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)),1))
        #yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)],dim=1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce1(torch.cat((yt_t,yt_f),1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1


        r1_prior_cam1 = -1 * (torch.sigmoid(x1_prior_cam)) + 1


        r2_prior_cam2 = -1 * (torch.sigmoid(x2_prior_cam)) + 1


        r_prior_cam = r_prior_cam + r1_prior_cam1+r2_prior_cam2

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out,out_edge], dim=1)
        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam + x2_prior_cam

        return y,edge
    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out
class DCI_decoder_3_noafs(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_3_noafs, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, x2, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        x2_prior_cam = F.interpolate(x2, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt = self.reduce(torch.cat((FI,prior_cam.expand(-1, X.size()[1], -1, -1),x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)),1))
        #yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)],dim=1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        #yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        #yt_out = self.reduce1(torch.cat((yt_t,yt_f),1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1


        r1_prior_cam1 = -1 * (torch.sigmoid(x1_prior_cam)) + 1


        r2_prior_cam2 = -1 * (torch.sigmoid(x2_prior_cam)) + 1


        r_prior_cam = r_prior_cam + r1_prior_cam1+r2_prior_cam2

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_t,out_edge], dim=1)
        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam + x2_prior_cam

        return y,edge
    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class DCI_decoder_3_noedge(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_3_noedge, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, x2, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        x2_prior_cam = F.interpolate(x2, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt = self.reduce(torch.cat((FI,prior_cam.expand(-1, X.size()[1], -1, -1),x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)),1))
        #yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)],dim=1))

        #out_edge = self.edge(yt)
        #edge = self.out_edge(out_edge)
        #edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce1(torch.cat((yt_t,yt_f),1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1


        r1_prior_cam1 = -1 * (torch.sigmoid(x1_prior_cam)) + 1


        r2_prior_cam2 = -1 * (torch.sigmoid(x2_prior_cam)) + 1


        r_prior_cam = r_prior_cam + r1_prior_cam1+r2_prior_cam2

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out], dim=1)
        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam + x2_prior_cam

        return y
    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class Module_3_4(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Module_3_4, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1), nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1), nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels // 2, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 5, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, x2,x3, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear', align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        x2_prior_cam = F.interpolate(x2, size=X.size()[2:], mode='bilinear', align_corners=True)
        x3_prior_cam = F.interpolate(x3, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt = self.reduce(torch.cat((FI, prior_cam.expand(-1, X.size()[1], -1, -1),
                                    x1_prior_cam.expand(-1, X.size()[1], -1, -1),
                                    x2_prior_cam.expand(-1, X.size()[1], -1, -1),x3_prior_cam.expand(-1, X.size()[1], -1, -1)), 1))
        # yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)],dim=1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce1(torch.cat((yt_t, yt_f), 1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1

        r1_prior_cam1 = -1 * (torch.sigmoid(x1_prior_cam)) + 1

        r2_prior_cam2 = -1 * (torch.sigmoid(x2_prior_cam)) + 1

        r2_prior_cam3 = -1 * (torch.sigmoid(x3_prior_cam)) + 1

        r_prior_cam = r_prior_cam + r1_prior_cam1 + r2_prior_cam2 + r2_prior_cam3

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out, out_edge], dim=1)
        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam + x2_prior_cam + x3_prior_cam
        return y, edge

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out


class DCI_decoder_4(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_4, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1,groups=in_channels), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, x2,x3, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear', align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        x2_prior_cam = F.interpolate(x2, size=X.size()[2:], mode='bilinear', align_corners=True)
        x3_prior_cam = F.interpolate(x3, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt = self.reduce(torch.cat((FI, prior_cam.expand(-1, X.size()[1], -1, -1),
                                    x1_prior_cam.expand(-1, X.size()[1], -1, -1),
                                    x2_prior_cam.expand(-1, X.size()[1], -1, -1),x3_prior_cam.expand(-1, X.size()[1], -1, -1)), 1))
        # yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)],dim=1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce1(torch.cat((yt_t, yt_f), 1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1

        r1_prior_cam1 = -1 * (torch.sigmoid(x1_prior_cam)) + 1

        r2_prior_cam2 = -1 * (torch.sigmoid(x2_prior_cam)) + 1

        r2_prior_cam3 = -1 * (torch.sigmoid(x3_prior_cam)) + 1

        r_prior_cam = r_prior_cam + r1_prior_cam1 + r2_prior_cam2 + r2_prior_cam3

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out, out_edge], dim=1)
        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam + x2_prior_cam + x3_prior_cam
        return y, edge

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out


class DCI_decoder_4_noedge(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_4_noedge, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1,groups=in_channels), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, x2,x3, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear', align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        x2_prior_cam = F.interpolate(x2, size=X.size()[2:], mode='bilinear', align_corners=True)
        x3_prior_cam = F.interpolate(x3, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt = self.reduce(torch.cat((FI, prior_cam.expand(-1, X.size()[1], -1, -1),
                                    x1_prior_cam.expand(-1, X.size()[1], -1, -1),
                                    x2_prior_cam.expand(-1, X.size()[1], -1, -1),x3_prior_cam.expand(-1, X.size()[1], -1, -1)), 1))
        # yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)],dim=1))

        #out_edge = self.edge(yt)
        #edge = self.out_edge(out_edge)
        #edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        yt_out = self.reduce1(torch.cat((yt_t, yt_f), 1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1

        r1_prior_cam1 = -1 * (torch.sigmoid(x1_prior_cam)) + 1

        r2_prior_cam2 = -1 * (torch.sigmoid(x2_prior_cam)) + 1

        r2_prior_cam3 = -1 * (torch.sigmoid(x3_prior_cam)) + 1

        r_prior_cam = r_prior_cam + r1_prior_cam1 + r2_prior_cam2 + r2_prior_cam3

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out], dim=1)
        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam + x2_prior_cam + x3_prior_cam
        return y

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out


class DCI_decoder_4_noafs(nn.Module):
    def __init__(self, in_channels):
        super(DCI_decoder_4_noafs, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1,groups=in_channels), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.joint_attention = AM(in_channels)

    def forward(self, X, x1, x2,x3, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear', align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        x2_prior_cam = F.interpolate(x2, size=X.size()[2:], mode='bilinear', align_corners=True)
        x3_prior_cam = F.interpolate(x3, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt = self.reduce(torch.cat((FI, prior_cam.expand(-1, X.size()[1], -1, -1),
                                    x1_prior_cam.expand(-1, X.size()[1], -1, -1),
                                    x2_prior_cam.expand(-1, X.size()[1], -1, -1),x3_prior_cam.expand(-1, X.size()[1], -1, -1)), 1))
        # yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)],dim=1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)+self.conv5(yt)
        #yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(yt.float()).real)))))
        #yt_out = self.reduce1(torch.cat((yt_t, yt_f), 1))

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1

        r1_prior_cam1 = -1 * (torch.sigmoid(x1_prior_cam)) + 1

        r2_prior_cam2 = -1 * (torch.sigmoid(x2_prior_cam)) + 1

        r2_prior_cam3 = -1 * (torch.sigmoid(x3_prior_cam)) + 1

        r_prior_cam = r_prior_cam + r1_prior_cam1 + r2_prior_cam2 + r2_prior_cam3

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_t, out_edge], dim=1)
        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam + x2_prior_cam + x3_prior_cam
        return y, edge

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ASPP, self).__init__()
        self.convert =nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )

        self.branch1_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),  nn.ReLU(True),
        )
        self.branch1_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=6, dilation=6), nn.ReLU(True),
        )
        self.branch1_4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=12, dilation=12),  nn.ReLU(True),
        )
        self.branch1_6 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=18, dilation=18),  nn.ReLU(True),
        )
        self.branch1_a = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channel, out_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(out_channel * 5, out_channel, 1, padding=0), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 1),
        )


    def forward(self, x):
        x0 = self.convert(x)
        x1 = self.branch1_1(x0)
        x2 = self.branch1_2(x0)
        x3 = self.branch1_4(x0)
        x4 = self.branch1_6(x0)
        x5 = self.branch1_a(x0)
        x5_ = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x_    = torch.cat((x1,x2,x3,x4,x5_),1)
        x = self.reduce1(x_)
        return x


class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.convert =nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )

        self.branch_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1),  nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3), nn.ReLU(True),
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5), nn.ReLU(True),
        )

        self.reduce1 = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, 1, padding=0), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 1),
        )


    def forward(self, x):
        x0 = self.convert(x)
        x1 = self.branch_1(x0)
        x2 = self.branch_2(x0)
        x3 = self.branch_3(x0)
        x  = self.reduce1(torch.cat((x1,x2,x3),1))
        return x


class DASPP(nn.Module):
    def __init__(self, inchannels, depth=128):

        super(DASPP, self).__init__()
        self.branch0 = nn.Sequential(nn.Conv2d(inchannels, depth, kernel_size=1, stride=1), nn.BatchNorm2d(depth),
                                     nn.ReLU(True))
        self.branch1 = nn.Sequential(
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=3, dilation=3), nn.BatchNorm2d(depth),
            nn.ReLU(True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(depth * 2, depth, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(depth), nn.ReLU(True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(depth * 3, depth, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(depth), nn.ReLU(True))
        self.branch4 = nn.Sequential(
            nn.Conv2d(depth * 4, depth, kernel_size=3, stride=1, padding=18, dilation=18),
            nn.BatchNorm2d(depth), nn.ReLU(True))
        self.branch5 = nn.Sequential(
            nn.Conv2d(depth * 5, depth, kernel_size=3, stride=1, padding=24, dilation=24),
            nn.BatchNorm2d(depth), nn.ReLU(True))

        self.head = nn.Sequential(
            nn.Conv2d(depth * 6, depth, kernel_size=1), nn.BatchNorm2d(depth), nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(depth, depth // 2, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch0 = self.branch0(x)
        branch1 = self.branch1(branch0)
        branch2 = self.branch2(torch.cat((branch0, branch1), 1))
        branch3 = self.branch3(torch.cat((branch0, branch1, branch2), 1))
        branch4 = self.branch4(torch.cat((branch0, branch1, branch2, branch3), 1))
        branch5 = self.branch5(torch.cat((branch0, branch1, branch2, branch3, branch4), 1))
        out = torch.cat([branch0, branch1, branch2, branch3, branch4, branch5], 1)
        out = self.head(out)
        out = self.out(out)
        return out






