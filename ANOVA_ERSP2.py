'''
Funcao ERDS
Autor: Éric Kauati Saito
Data: 26/10/2021
'''

import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft
from scipy import stats
import scipy.io as spio
import my_eeg_fnc as my
import pywt


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def my_filter(x, fs, order, freq1, freq2):
    w1 = freq1 / (fs / 2)  # Normalize the frequency
    [b1, a1] = signal.butter(order, w1, btype='high')
    x_aux = signal.filtfilt(b1, a1, x)
    w2 = freq2 / (fs / 2)
    [b2, a2] = signal.butter(order, w2, btype='low')
    x_out = signal.filtfilt(b2, a2, x_aux)
    return x_out

'''
def erds(eeg, fs, f1, f2, t1=-4, t2=4):
    n_vol = len(eeg)
    n_ch = np.size(eeg[0], 1)
    n = np.size(eeg[0], 2)  # n amostras

    order = 4 # Ordem do filtro

    eeg_filt = []
    erp_aux = np.zeros((n_vol, n_ch, n))

    # ERP
    for ind in range(n_vol):
        erp_aux[ind, :, :] = np.mean(eeg[ind], axis=0)
    erp = np.mean(erp_aux, axis=0)

    # ERDS - Subtração do ERP, Filtragem por Banda
    for ind in range(n_vol):
        eeg_vol = eeg[ind]
        for ind_epoca in range(np.size(eeg_vol, 0)):
            eeg_erp = eeg_vol[ind_epoca, :, :] - erp[:, :]  # -ERP
            eeg_aux = my.my_filter(eeg_erp, fs, order, f1, f2)
            eeg_filt.append(eeg_aux)
    eeg_filt = np.array(eeg_filt)

    # ERDS - Calculo
    w_size = 0.5
    noverlap = 0.5
    ref = 1

    ERDS_w = []
    ERDS_N = []
    P_N = []
    for ind_ch in range(n_ch):
        eeg_ch = eeg_filt[:, ind_ch, :-1]
        eeg_pwr = eeg_ch ** 2
        P_0 = np.mean(eeg_pwr, 0)  # Media das épocas
        R = np.mean(P_0[0:int(ref * fs)])
        P_w = strided_app(P_0, int(w_size * fs), int(noverlap * w_size * fs))
        P = np.mean(P_w, 1)
        ERDS_P = ((P - R) / R) * 100
        ERDS_aux = ((P_0 - R) / R) * 100
        ERDS_w.append(ERDS_P)
        ERDS_N.append(ERDS_aux)
        P_N.append(P_0)

    # ERDS - Filtragem(Moving Average)
    w = 3
    ERDS_MA = []
    ERDSN_N_MA = []
    P_N_MA = []
    for ind_ch in range(n_ch):
        ERDS_MA.append(np.convolve(ERDS_w[ind_ch], np.ones(w) / w, 'valid'))
        ERDSN_N_MA.append(np.convolve(ERDS_N[ind_ch], np.ones(w) / w, 'valid'))
        P_N_MA.append(np.convolve(P_N[ind_ch], np.ones(w) / w, 'valid'))

    t = strided_app(np.linspace(t1, t2, np.size(P_0)), int(w_size * fs), int(noverlap * w_size * fs))
    t_teste = np.convolve(t[:, 0], np.ones(w) / w, 'valid')
    t_erds = t[:, 0]

    return ERDS_MA, t_erds, ERDSN_N_MA, P_N_MA, t_teste
'''

MV_epocas = []
IM_epocas = []
vol = 6

for i in range(vol):
    load_path = r'D:\Doutorado\Arquivos dos Sinais\Sinais Processados 2\Alexsandro\Voluntario '
    atividade_MV = r'\artefato rejeitado\MVvol'
    epocas_MV = mne.read_epochs(load_path + str(i + 1) + atividade_MV + str(i + 1) + '_ICA-epo.fif')
    MV_epocas.append(epocas_MV.get_data(picks='eeg'))

    atividade_IM = r'\artefato rejeitado\IMvol'
    epocas_IM = mne.read_epochs(load_path + str(i + 1) + atividade_IM + str(i + 1) + '_ICA-epo.fif')
    IM_epocas.append(epocas_IM.get_data(picks='eeg'))

for i in range(vol):
    load_path = r'D:\Doutorado\Arquivos dos Sinais\Sinais Processados 2\Ernesto Lana\Voluntario '
    atividade_MV = r'\artefato rejeitado\MVvol'
    epocas_MV = mne.read_epochs(load_path + str(i + 1) + atividade_MV + str(i + 1) + '_1_ICA-epo.fif')
    MV_epocas.append(epocas_MV.get_data(picks='eeg'))
    epocas_MV = mne.read_epochs(load_path + str(i + 1) + atividade_MV + str(i + 1) + '_2_ICA-epo.fif')
    MV_epocas.append(epocas_MV.get_data(picks='eeg'))

    atividade_IM = r'\artefato rejeitado\IMvol'
    epocas_IM = mne.read_epochs(load_path + str(i + 1) + atividade_IM + str(i + 1) + '_1_ICA-epo.fif')
    IM_epocas.append(epocas_IM.get_data(picks='eeg'))
    epocas_IM = mne.read_epochs(load_path + str(i + 1) + atividade_IM + str(i + 1) + '_2_ICA-epo.fif')
    IM_epocas.append(epocas_IM.get_data(picks='eeg'))


ch_name = epocas_MV.ch_names
fs = epocas_MV.info['sfreq']

#Deleta Oz dos dados do alexsandro
for ind in range(vol):
    IM_epocas[ind] = np.delete(IM_epocas[ind],17,axis=1)
    MV_epocas[ind] = np.delete(MV_epocas[ind],17,axis=1)


del epocas_MV, epocas_IM

f1 = 8
f2 = 13
eeg = IM_epocas
n_vol = len(eeg)
n_ch = np.size(eeg[0], 1)
n = np.size(eeg[0], 2)  # n amostras

order = 4 # Ordem do filtro

eeg_filt = []
erp_aux = np.zeros((n_vol, n_ch, n))
erp2_aux = np.zeros((n_vol, n_ch, n))
# ERP
for ind in range(n_vol):
    erp_aux[ind, :, :] = np.mean(eeg[ind], axis=0)
    erp2_aux[ind, :, :] = np.mean(eeg[ind]**2, axis=0)
erp = np.mean(erp_aux, axis=0)
erp2 = np.mean(erp2_aux, axis=0)

# ERDS - Subtração do ERP, Filtragem por Banda
for ind in range(n_vol):
    eeg_vol = eeg[ind]
    for ind_epoca in range(np.size(eeg_vol, 0)):
        eeg_erp = eeg_vol[ind_epoca, :, :] - erp[:, :]  # -ERP
        eeg_aux = my.my_filter(eeg_erp, fs, order, f1, f2)
        eeg_filt.append(eeg_aux)
eeg_filt = np.array(eeg_filt)


# ERDS - Calculo
w_size = 0.5
noverlap = 0.5
ref = 1

ERDS_w = []
ERDS_N = []
P_N = []
eeg_pwr_N = []
for ind_ch in range(n_ch):
    eeg_ch = eeg_filt[:, ind_ch, :-1]
    eeg_pwr = eeg_ch ** 2
    P_0 = np.mean(eeg_pwr, 0)  # Media das épocas
    R = np.mean(P_0[0:int(ref * fs)])
    P_w = strided_app(P_0, int(w_size * fs), int(noverlap * w_size * fs))
    P = np.mean(P_w, 1)
    ERDS_P = ((P - R) / R) * 100
    ERDS_aux = ((P_0 - R) / R) * 100
    ERDS_w.append(ERDS_P)
    ERDS_N.append(ERDS_aux)
    P_N.append(P_0)
    eeg_pwr_N.append(eeg_pwr)

# ERDS - Filtragem(Moving Average)
w = 3
ERDS_MA = []
ERDSN_N_MA = []
P_N_MA = []
for ind_ch in range(n_ch):
    ERDS_MA.append(np.convolve(ERDS_w[ind_ch], np.ones(w) / w, 'valid'))
    ERDSN_N_MA.append(np.convolve(ERDS_N[ind_ch], np.ones(w) / w, 'valid'))
    P_N_MA.append(np.convolve(P_N[ind_ch], np.ones(w) / w, 'valid'))

t = strided_app(np.linspace(-4, 4, np.size(P_0)), int(w_size * fs), int(noverlap * w_size * fs))
t_teste = np.convolve(t[:, 0], np.ones(w) / w, 'valid')
t_erds = t[:, 0]
'''    
t = np.linspace(-4, 4, np.size(ERDS_N_MA[0]))
fig, axes = plt.subplots(nrows=4, ncols=5, sharey=True)
axes = axes.flatten()
ch_v = [0, 5, 10, 1, 6, 11, 16, 4, 9, 14, 3, 8, 13, 18, 2, 7, 12, 17] #Alexsandro
ylim = [-70, 70]
ind = 0
fig.subplots_adjust(wspace=0.1, hspace=0.5)
major_ticks_top = np.linspace(-4, 3, 8)
for ind_ch in ch_v:
    plt.suptitle('MV/IM (' + str(f1) + '-' + str(f2) + ' Hz)')
    l1 = axes[ind_ch].plot(t, ERDS_N_MA[ind])
    l2 = axes[ind_ch].plot(t_erds[:-2], ERDS_MA[ind])
    l3 = axes[ind_ch].plot(t, P_N_MA[ind])
    axes[ind_ch].set_xticks(major_ticks_top)
    axes[ind_ch].axvline(x=0, linewidth=1, color='r', linestyle="--")
    axes[ind_ch].axvline(x=2.5, linewidth=1, color='g', linestyle="--")
    axes[ind_ch].set_title(ch_name[ind])
    axes[ind_ch].set_ylim(ylim)
    ind += 1

fig.delaxes(axes[15])
fig.delaxes(axes[17])  # Lana
fig.delaxes(axes[19])

plt.show()
'''
teste = eeg_pwr_N
#t = np.linspace(-4, 4, np.size(teste[0][0,:]))
fig, axes = plt.subplots(nrows=4, ncols=5, sharey=True)
axes = axes.flatten()
ch_v = [0, 5, 10, 1, 6, 11, 16, 4, 9, 14, 3, 8, 13, 18, 2, 7, 12] #Alexsandro
ind = 0
fig.subplots_adjust(wspace=0.1, hspace=0.5)
major_ticks_top = np.linspace(-4, 3, 8)
for ind_ch in ch_v:
    plt.suptitle('MV/IM (' + str(f1) + '-' + str(f2) + ' Hz)')
    l3 = axes[ind_ch].plot(teste[ind][0,:])
    #axes[ind_ch].set_xticks(major_ticks_top)
    #axes[ind_ch].axvline(x=0, linewidth=1, color='r', linestyle="--")
    #axes[ind_ch].axvline(x=2.5, linewidth=1, color='g', linestyle="--")
    axes[ind_ch].set_title(ch_name[ind])
    ind += 1
fig.delaxes(axes[15])
fig.delaxes(axes[17])  # Lana
fig.delaxes(axes[19])

plt.show()

t = np.linspace(-4, 4, np.size(ERDSN_N_MA[0]))
t1 = np.where((t>0) & (t<1))

from scipy import stats

C3_t1 = ERDSN_N_MA[4][t1]
C4_t1 = ERDSN_N_MA[11][t1]

# Qual canal tem maior ERDS (par)
stats.ttest_rel(C3_t1,C4_t1) # Indica se são iguais

# TESTAR COM DADOS SIMULADOS
# stats.ks_2samp(C3_t1, C4_t1, alternative="less") # two-sample Kolmogorov-Smirnov
print('C3-C4')
print(stats.mannwhitneyu(C3_t1, C4_t1))
print(stats.wilcoxon(C3_t1, C4_t1))
#stats.mannwhitneyu(C3_t1, C4_t1, alternative="less") # Mann-Whitney U rank test on two independent samples
# F(x) < G(x)
#stats.wilcoxon(C3_t1, C4_t1, alternative="less") # Wilcoxon signed-rank test
# F(x) < G(x)


P3_t1 = ERDSN_N_MA[5][t1]
P4_t1 = ERDSN_N_MA[12][t1]

print('P3-P4')
print(stats.mannwhitneyu(P3_t1, P4_t1))
print(stats.wilcoxon(P3_t1, P4_t1))

#stats.mannwhitneyu(P3_t1, P4_t1, alternative="less")
# F(x) < G(x)
#stats.wilcoxon(P3_t1, P4_t1, alternative="less")
# F(x) < G(x)

F3_t1 = ERDSN_N_MA[3][t1]
F4_t1 = ERDSN_N_MA[10][t1]

print('F3-F4')
print(stats.mannwhitneyu(F3_t1, F4_t1))
print(stats.wilcoxon(F3_t1, F4_t1))

#stats.mannwhitneyu(F3_t1, F4_t1, alternative="less")
# F(x) < G(x)
#stats.wilcoxon(F3_t1, F4_t1, alternative="less")
# F(x) < G(x)


canais = [C3_t1, C4_t1, P3_t1, P4_t1, F3_t1, F4_t1]
nomes = ['C3', 'C4', 'P3', 'P4', 'F3', 'F4']

plt.figure()
plt.suptitle('IM (' + str(f1) + '-' + str(f2) + ' Hz)')
plt.boxplot(canais, labels=nomes)
x1, x2 = 1, 2
y, h, col = np.max([C3_t1, C4_t1]) + 2, 2, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

x1, x2 = 3, 4
y, h, col = np.max([P3_t1, P4_t1]) + 2, 2, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

x1, x2 = 5, 6
y, h, col = np.max([F3_t1, F4_t1]) + 2, 2, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
