
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import scipy.io as spio
import my_eeg_fnc as my
from my_sobi import my_ica, ICA_comp_plot, recon_plot
import matplotlib
matplotlib.use("Qt5Agg")
# LoadData Alexsandro
load_path = r'D:\Mestrado\Arquivos dos Sinais\Alexsandro\\'
coleta = 'IMvol1.mat'
data_load = spio.loadmat(load_path + coleta, squeeze_me=True)
data0 = data_load['IMvol1']
data0 = data0.transpose()
data0 = data0[:, :-1]
fs = data_load['Fs']
ch_name = data_load['canal']
ch_name = ch_name.tolist()
n_eeg = 20
ch_acc = 25
# Info Alexsandro
ch_types = ['eeg'] * 20 + ['misc'] * 5 + ['stim'] + ['misc']

# Eventos retirado do acelerometro
data = my.my_epoca(data0, fs, 0.1, 0.05, ch_acc, False)
eeg = data[:n_eeg, :]
A1A2 = data[n_eeg, :]
misc = data[n_eeg+1:, :]
data_raw = np.zeros(data.shape)
data_raw[data_raw == 0] = np.nan

# Filtragem
eeg_filtrado0 = my.my_notch(eeg, fs, 60, 15)
eeg_filtrado1 = my.my_notch(eeg_filtrado0, fs, 180, 15)
eeg_filtrado = my.my_filter(eeg_filtrado1, fs, 2, 1, 45)

A1A2_filtrado0 = my.my_notch(A1A2, fs, 60, 15)
A1A2_filtrado1 = my.my_notch(A1A2_filtrado0, fs, 180, 15)
A1A2_filtrado = my.my_filter(A1A2_filtrado1, fs, 2, 1, 45)

del eeg_filtrado0, eeg_filtrado1, A1A2_filtrado0, A1A2_filtrado1

# EoG Alexsandro
eog = (eeg_filtrado[3, :] + eeg_filtrado[11, :]) / 2
w1 = 10 / (fs / 2)  # Normalize the frequency
[b1, a1] = signal.butter(2, w1, btype='low')
eog_filtrado = signal.filtfilt(b1, a1, eog)

ch_name += ['EOG']
ch_types += ['eog']
data_raw = np.vstack((data_raw, np.zeros((1, len(eog))) + np.nan))
#
data_raw[:n_eeg, :] = eeg_filtrado
data_raw[n_eeg, :] = A1A2_filtrado
data_raw[n_eeg+1:-1, :] = misc
data_raw[-1, :] = eog_filtrado
A1A2_0 = A1A2_filtrado[ 3 * fs:]

data_raw2 = data_raw[:, 3 * fs:]  # Alexsandro-problema nos primeiros 3 segundos

del eeg_filtrado, data_raw, eeg, data0, eog, data, misc, eog_filtrado, A1A2

# Montage 10-20
info = mne.create_info(ch_name, ch_types=ch_types, sfreq=fs)
raw = mne.io.RawArray(data_raw2, info)
raw.info['lowpass'] = 45.0
raw.info['highpass'] = 1.0
raw = my.my_montage(raw, ch_name[:21])  # Alexsandro
raw.drop_channels(['A1-2'])

A1A2_info = mne.create_info(['A1-A2'], raw.info['sfreq'], ['eeg'])
A1A2_raw = mne.io.RawArray(A1A2_0.reshape((1, len(A1A2_0))), A1A2_info)
raw.add_channels([A1A2_raw], force_update_info=True)

# ICA
eeg_recon, S, corrp = my_ica(raw)

recon_raw = raw.copy()
recon_raw['eeg'] = eeg_recon


# Epocas
events = mne.find_events(raw, consecutive=True)
event_dict = {'flex': 1, 'ext': 2}
order = np.append(np.arange(20), 20)
escala = dict(eeg=80e-6)


#reject_criteria = dict(eeg=100e-6)
epochs = mne.Epochs(recon_raw, events, event_dict, tmin=0, tmax=4, baseline=(None, None),
                    reject_tmin=None, reject_tmax=4)
epochs.plot(picks=['eeg', 'eog'], events=events, event_id=event_dict, scalings=escala, n_epochs=5, n_channels=4)

epoch_eeg = epochs.get_data(picks='eeg')

#
#a = [int(x) for x in input().split()]

epoch_base = np.array([1, 2, 3, 4, 5])
'''
m_dp = 3
m_percent = 5
m_percent_total = 10
amp = 150e-6
signal_out, signal_discarded, epoch_discarded = my.rejeitaartefato(
    epoch_eeg, epoch_base, m_dp, m_percent, m_percent_total, amp)

mask = np.ones(epoch_eeg.shape[0], dtype=bool)
mask[epoch_discarded] = False
epoch_teste = epochs[mask]
'''