import pywt
import pywt.data
import torch
import torch.nn.functional as F


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)


    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    print(dec_filters.shape)
    print(in_size)
    

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    print("dec_filters shape:", dec_filters.shape)
    print("rec_filters shape:", rec_filters.shape)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    print(x.shape)
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)#([2, 12, 32, 32])
    x = x.reshape(b, c, 4, h // 2, w // 2)#torch.Size([2, 3, 4, 32, 32])
    
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    #print('啦',x.shape)#torch.Size([2, 3, 4, 32, 32])
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
