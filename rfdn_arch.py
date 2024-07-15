
from basicsr.utils.registry import ARCH_REGISTRY
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F





def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'gelu':
        layer = nn.GELU()

    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer



class Partial_conv3(nn.Module):

    def __init__(self, dim, kernel_size=3,stride=1,n_div=4,forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3#idenity
        self.partial_conv3 = conv_layer(self.dim_conv3,self.dim_conv3,kernel_size=kernel_size,stride=stride,)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

class BSConvU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode='reflect',
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class LocalAwareAttention(nn.Module):
    '''
        Local Aware Attention Module
    '''

    def __init__(self, kernel_size=4, stride=4, beta=0.07):
        super(LocalAwareAttention, self).__init__()
        # Module Network Parameters
        self.beta = beta
        self.kernel_size = kernel_size
        self.stride = stride
        self.scale_factor = kernel_size
        # Layers
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)
        #self.upsample = nn.Upsample(scale_factor=self.scale_factor)
    def forward(self, x):
        # print('Input-Shape', x.shape)
        avg_pool = self.avg_pool(x)
        # print('AVG_Pool', avg_pool.shape)
        #upsample = self.upsample(avg_pool)
        upsample = F.interpolate(avg_pool, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        # print('UpSample2d', upsample.shape)

        sub_relu = self.beta * F.relu(torch.sub(x, upsample))
        # print('Sub Relu', sub_relu.shape)
        mul_op = torch.mul(sub_relu, x)
        # print('Mul OP', mul_op.shape)

        op = torch.add(x, mul_op)
        # print('OP', op.shape)

        return op





def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)




def sequential(*args):
    '''if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]'''
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)



def get_local_weights(residual, ksize, padding):
    pad = padding
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
    unfolded_residual = residual_pad.unfold(2, ksize, 3).unfold(3, ksize, 3)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight

class ESA(nn.Module):
    def __init__(self, num_feat, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = num_feat // 4
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.conv2_0 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_1 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_2 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_3 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.maxPooling_0 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.maxPooling_1 = nn.MaxPool2d(kernel_size=5, stride=3)
        self.maxPooling_2 = nn.MaxPool2d(kernel_size=7, stride=3, padding=1)
        self.conv_max_0 = Partial_conv3(f,kernel_size=3,n_div=4)
        self.conv_max_1 = Partial_conv3(f,kernel_size=3,n_div=4)
        self.conv_max_2 = Partial_conv3(f,kernel_size=3,n_div=4)
        self.var_3 = get_local_weights

        self.conv3_0 = Partial_conv3(f,kernel_size=3,n_div=4)
        self.conv3_1 = Partial_conv3(f,kernel_size=3,n_div=4)
        self.conv3_2 = Partial_conv3(f,kernel_size=3,n_div=4)

        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()
        #self.pa = PA(f)

    def forward(self, input):
        c1_ = self.conv1(input)  # channel squeeze
        temp = self.conv2_0(c1_)
        c1_0 = self.maxPooling_0(temp)  # strided conv 3
        c1_1 = self.maxPooling_1(self.conv2_1(c1_))  # strided conv 5
        c1_2 = self.maxPooling_2(self.conv2_2(c1_))  # strided conv 7
        c1_3 = self.var_3(self.conv2_3(c1_), 7, padding=1)  # strided local-var 7


        v_range_0 = self.conv3_0(self.GELU(self.conv_max_0(c1_0)))
        v_range_1 = self.conv3_1(self.GELU(self.conv_max_1(c1_1)))
        v_range_2 = self.conv3_2(self.GELU(self.conv_max_2(c1_2 + c1_3)))


        c3_0 = F.interpolate(v_range_0, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        c3_1 = F.interpolate(v_range_1, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        c3_2 = F.interpolate(v_range_2, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)

        #c4 = self.conv4((c3_0 + c3_1 + c3_2 + self.pa(self.conv_f(c1_))))
        c4 = self.conv4((c3_0 + c3_1 + c3_2 + c1_))
        m = self.sigmoid(c4)
        output = input * m

        return output+input
'''
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv_layer(n_feats, f, kernel_size=1)
        self.conv_f = conv_layer(f, f, kernel_size=1)
        self.conv_max = conv(f, f,kernel_size=3)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f,kernel_size=3)
        self.conv3_ = conv_layer(f, f,kernel_size=3)
        self.conv4 = conv_layer(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.conv_max(v_max)
        c3 = self.conv3(v_range)
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.pa = PA(f)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        cf = self.pa(cf)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        return x * m'''


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CCALayer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)+self.contrast(x)
        y = self.conv_du(y)
        output = x * y
        return output

class PCB(nn.Module):
    def __init__(self,channel,n_div=4):
        super(PCB, self).__init__()
        self.partialconv = Partial_conv3(channel,kernel_size=3,n_div=n_div)
        self.gelu = nn.GELU()
        self.pw1 = conv_layer(channel,channel*2,1)
        self.pw2 = conv_layer(channel*2,channel,1)

    def forward(self,x):
        y = self.pw2(self.gelu(self.pw1(self.partialconv(x))))

        return  x + y


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = Partial_conv3(in_channels, 3,n_div=1)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = Partial_conv3(self.remaining_channels,  3, n_div=2)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = Partial_conv3(self.remaining_channels,  3, n_div=4)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = nn.GELU()
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.ESA = ESA(in_channels, nn.Conv2d)
        self.cca = CCALayer(in_channels)



    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        out = self.cca(out)
        out_fused = self.ESA(out)+input

        return out_fused

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


@ARCH_REGISTRY.register()
class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=5, out_nc=3, upscale=2,rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(RFDN, self).__init__()

        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.feature = conv_layer(in_nc, nf,kernel_size=3)
        self.laa = LocalAwareAttention()
        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.B5 = RFDB(in_channels=nf)
        #self.B6 = RFDB(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='gelu')
        self.LR_conv = Partial_conv3(nf, forward='split_cat')
        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0
        self.c1 = conv_layer(nf,nf,3)
        self.pa = PA(nf)

    def forward(self, input):
        self.mean = self.mean.type_as(input)
        input = input - self.mean

        out_fea = self.feature(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        #out_B6 = self.B6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5], dim=1))
        out_B = self.pa(out_B)
        out_lr = self.LR_conv(out_B)+out_fea
        output = self.upsampler(out_lr)+self.upsampler(self.c1(out_fea))+self.mean

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx