import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from einops import rearrange
import numbers

def channel_shuffle(x, groups=4):

    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class MSF(nn.Module):
    def __init__(self, in_channels):
        super(MSF, self).__init__()

        self.Dconv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=3,dilation=3), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU(True)
        )
        self.Dconv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=5,dilation=5), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=7,dilation=7), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv9 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=9,dilation=9), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv12 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=12, dilation=12), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels * 3, 1), nn.BatchNorm2d(in_channels * 3),
            nn.Conv2d(in_channels * 3, in_channels * 2, 3, padding=1), nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

    def forward(self, F1):
       F1_3 = self.Dconv3(F1)
       F1_5 = self.Dconv5(F1+F1_3)
       F1_7 = self.Dconv7(F1+F1_5)
       F1_9 = self.Dconv9(F1+F1_7)
       F1_12 = self.Dconv12(F1+F1_9)
       out = self.out(torch.cat((F1_3,F1_5,F1_7,F1_9,F1_12),1)) + F1

       return out

    def initialize(self):
        weight_init(self)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        self.dwconv0 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(dim)
        self.project_in  = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.joint_attention = AM(dim)

    def forward(self, x):
        x    = self.project_in(x)
        x_s1 = self.dwconv0(x)
        x_s    = F.gelu(x_s1)*x_s1
        x_s    = self.dwconv1(x_s)
        x_f    = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(x).real)))))

        out = self.project_out(torch.cat((x_f,x_s),1))

        return out

    def initialize(self):
        weight_init(self)



def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor


class Attention_G(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_G, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.PConv1 = nn.Conv2d(dim,dim,kernel_size=1)
        self.PConv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.PConv3 = nn.Conv2d(dim, dim, kernel_size=1)
        self.PConv_1 = nn.Conv2d(dim, dim, kernel_size=1)

        self.DWConv3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.DWConv3_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.DWConv3_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)


        self.joint_attention = AM(dim)
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(dim)


    def forward(self, x):
        b, c, h, w = x.shape

        q_s = self.DWConv3_1(self.PConv1(x))
        k_s = self.DWConv3_2(self.PConv2(x))
        v_s = self.DWConv3_3(self.PConv3(x))

        out_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(self.PConv_1(x).float()).real)))))

        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = attn_s.softmax(dim=-1)
        out_s = (attn_s @ v_s)
        out_s = rearrange(out_s, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out   = self.project_out(torch.cat((out_s,out_f),1))

        return out

    def initialize(self):
        weight_init(self)

class Attention_sam(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_sam, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.PConv1 = nn.Conv2d(dim,dim,kernel_size=1)
        self.PConv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.PConv3 = nn.Conv2d(dim, dim, kernel_size=1)
        self.PConv_1 = nn.Conv2d(dim, dim, kernel_size=1)

        self.DWConv3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.DWConv3_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.DWConv3_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


        #self.joint_attention = AM(dim)
        #self.norm = nn.BatchNorm2d(dim)
        #self.relu = nn.ReLU(dim)


    def forward(self, x):
        b, c, h, w = x.shape

        q_s = self.DWConv3_1(self.PConv1(x))
        k_s = self.DWConv3_2(self.PConv2(x))
        v_s = self.DWConv3_3(self.PConv3(x))

        #out_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.joint_attention(torch.fft.fft2(self.PConv_1(x).float()).real)))))

        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = attn_s.softmax(dim=-1)
        out_s = (attn_s @ v_s)
        out_s = rearrange(out_s, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out   = self.project_out(out_s)

        return out

    def initialize(self):
        weight_init(self)
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

class HIM(nn.Module):
    def __init__(self, channels):
        super(HIM, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(channels * 4, channels*2, kernel_size=1),nn.BatchNorm2d(channels*2),
            nn.Conv2d(channels * 2, channels*2, kernel_size=3,padding=1), nn.BatchNorm2d(channels*2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(True)
        )
        self.JA  = AM(channels)

    def forward(self, x_1,x_2):
        x_m = x_1*x_2
        x_a = x_1+x_2
        x_c = self.out(torch.cat((x_m,x_a,x_1,x_2),1))
        x = self.JA(x_c)
        return x

class HIM3(nn.Module):
    def __init__(self, channels):
        super(HIM3, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(channels * 5, channels * 3, kernel_size=1), nn.BatchNorm2d(channels * 3),
            nn.Conv2d(channels * 3, channels * 2, kernel_size=3, padding=1), nn.BatchNorm2d(channels * 2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(True)
        )
        self.JA  = AM(channels)

    def forward(self, x_1,x_2,x_3):
        x_m = x_1*x_2*x_3
        x_a = x_1+x_2+x_3
        x_c = self.out(torch.cat((x_m,x_a,x_1,x_2,x_3),1))
        x = self.JA(x_c)
        return x

class HIM4(nn.Module):
    def __init__(self, channels):
        super(HIM4, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(channels * 6, channels * 3, kernel_size=1), nn.BatchNorm2d(channels*3),
            nn.Conv2d(channels * 3, channels * 2, kernel_size=3, padding=1), nn.BatchNorm2d(channels*2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(True)
        )
        self.JA  = AM(channels)

    def forward(self, x_1,x_2,x_3,x_4):
        x_m = x_1*x_2*x_3*x_4
        x_a = x_1+x_2+x_3+x_4
        x_c = self.out(torch.cat((x_m,x_a,x_1,x_2,x_3,x_4),1))
        x = self.JA(x_c)
        return x
class Module_1(nn.Module):
    def __init__(self, dim=128, num_heads=8, ffn_expansion_factor=4, bias=False,LayerNorm_type='WithBias'):
        super(Module_1, self).__init__()
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_G = Attention_G(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn_G(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x





    def initialize(self):
        weight_init(self)

class Module_1_sam(nn.Module):
    def __init__(self, dim=128, num_heads=8, ffn_expansion_factor=4, bias=False,LayerNorm_type='WithBias'):
        super(Module_1_sam, self).__init__()
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_G = Attention_sam(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn_G(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x





    def initialize(self):
        weight_init(self)

class Module1_res(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Module1_res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True))
        self.res_3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )
        self.res_5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )

        self.reduce  = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 1),nn.BatchNorm2d(out_channel), nn.ReLU(True))
        self.module1 = Module_1(dim=out_channel)

    def forward(self, x):
        x0    = self.conv1(x)
        x_FT  = self.module1(x0)
        x_local = self.res_3(x)+self.res_5(x)
        x     = self.reduce(torch.cat((x_local,x_FT),1)) + x0
        return x



class SNP_module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SNP_module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True) )

        self.res_3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1, groups= out_channel), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )

        self.res_5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1, groups= out_channel), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )

        self.reduce  = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 1),nn.BatchNorm2d(out_channel), nn.ReLU(True))

        self.module1 = Module_1(dim=out_channel)

    def forward(self, x):
        x0    = self.conv1(x)
        x_FT  = self.module1(x0)
        x_local = self.res_3(x0)+self.res_5(x0)
        x     = self.reduce(torch.cat((x_local,x_FT),1)) + x0
        return x



class SNP_module_lb_sam(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SNP_module_lb_sam, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True) )

        self.res_3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1, groups= out_channel), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )

        self.res_5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1, groups= out_channel), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )

        self.reduce  = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 1),nn.BatchNorm2d(out_channel), nn.ReLU(True))

        self.module1 = Module_1_sam(dim=out_channel)

    def forward(self, x):
        x0    = self.conv1(x)
        x_FT  = self.module1(x0)
        x_local = self.res_3(x0)+self.res_5(x0)
        x     = self.reduce(torch.cat((x_local,x_FT),1)) + x0
        return x



class SNP_module_LB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SNP_module_LB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True) )

        self.res_3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1, groups= out_channel), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )

        self.res_5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1, groups= out_channel), nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )

    def forward(self, x):
        x0    = self.conv1(x)
        #x_FT  = self.module1(x0)
        x_local = self.res_3(x0)+self.res_5(x0) +x0
        #x     = self.reduce(torch.cat((x_local,x_FT),1)) + x0
        return x_local





















