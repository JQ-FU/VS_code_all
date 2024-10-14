import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .util import wavelet


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        #创建正向，逆向小波滤波器
        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
         

        #滤波器设置为不可训练参数
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        #将 wavelet_transform 函数和 self.wt_filter（小波分解滤波器）绑定。当模型在前向传播中需要对输入进行小波分解时，可以直接调用 self.wt_function，而不需要每次传入滤波器
        self.wt_function = partial(wavelet.wavelet_transform, filters = self.wt_filter)
        self.iwt_function = partial(wavelet.inverse_wavelet_transform, filters = self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])#对输入张量的每个通道进行按元素的缩放

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []#储存低频部分
        x_h_in_levels = []#储存高频部分
        shapes_in_levels = []#储存张量形状

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            #小波变换要求张量尺寸是偶数，这一步将奇数填充到偶数
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)#小波变换
            curr_x_ll = curr_x[:,:,0,:,:]#提取低频部分（curr_x[:,:,0,:,:]一般表示第1个通道，一般低频）
            
            shape_x = curr_x.shape#获取当前小波变换的形状
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])#通过 reshape 将 curr_x 的形状修改为 (batch_size, channels * 4, height, width)
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))#卷积，缩放处理
            curr_x_tag = curr_x_tag.reshape(shape_x)#重塑为原形状

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])#将当前处理后的低频部分添加到 x_ll_in_levels 列表中。
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])#将高频部分（通常是通道1到3的输出）添加到 x_h_in_levels 列表中

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)



import torch
from torchsummary import summary

# 定义模型参数
in_channels = 3  # 输入通道数
out_channels = 3  # 输出通道数
kernel_size = 5
stride = 1
wt_levels = 1  # 小波分解的层数
wt_type = 'db1'  # 小波类型

# 实例化模型
model = WTConv2d(in_channels, out_channels, kernel_size, stride, wt_levels=wt_levels, wt_type=wt_type)

# 定义输入张量的形状
input_shape = (1, in_channels, 64, 64)  # 批次大小为1，通道数为in_channels，64x64的输入图像

# 使用summary函数查看模型结构
summary(model, input_size=input_shape[1:], device='cpu')

