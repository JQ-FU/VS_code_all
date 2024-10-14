import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.preprocessing import ICA
#import mnelab
import matplotlib.dates as mdate

data_path=r'C:\\Users\\FJQ\Desktop\sub01-05\sub01-05\sub01\\t1_task_convert.cdt.edf'
raw=mne.io.read_raw_edf(data_path,preload=True,exclude=('EGG','ECG'))
chan_types_dict={raw.info.ch_names[i]:'eeg' for i in range(32)}
chan_types_dict={'VEOG':'eog','HEOG':'eog'}
raw.set_channel_types(chan_types_dict)
#重命名原始数据中的通道名称
raw.rename_channels(mapping={'F11':'F9','F12':'F10','FT11':'FT9','FT12':'FT10'})
raw.info.set_montage('standard_1020')
montage=mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage,on_missing='raise',verbose=None)

#原始数据
raw.plot(duration=5,n_channels=32,clipping=None)
raw.plot_psd(average=True)

#重参考
raw_bip_ref=mne.set_bipolar_reference(raw,anode=['M1'],cathode=['M2'])

#滤波
raw=raw.notch_filter(freqs=(50))
raw=raw.filter(l_freq=0.5,h_freq=100)
raw.plot_psd(average=True)
raw.info

#ICA
ica=ICA(n_components=32,max_iter='auto')#n_components=int根据实际需求写
raw_for_ica=raw.copy().filter(l_freq=1,h_freq=None)
ica.fit(raw_for_ica)
ica.plot_sources(raw_for_ica)
ica.plot_components()

ica.plot_overlay(raw_for_ica,exclude=[1])

ica.exclude=[1]
ica.apply(raw)

#降采样
raw = raw.resample(sfreq=200)
raw.set_eeg_reference(ref_channels='average')

#截止时间
raw.crop(tmin=0,tmax=300).load_data()
raw.pick_types(eeg=True, stim=False).plot(duration=5,n_channels=32,clipping=None)

import pathlib
raw.save(pathlib.Path('C:111') / 't1rest2.raw.fif', overwrite=True)