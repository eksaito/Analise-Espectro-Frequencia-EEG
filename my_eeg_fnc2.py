
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft
import scipy.io as spio
import my_eeg_fnc as my
import pywt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy import stats
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker

'''
My functions to work with EEG
10/12/2021
    -erds
    -erp
    -plot
    -montage
    -topographic map
'''

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def erds(eeg, t1, t2, freq1, freq2, fs):
    # ERDS vol não tem filtro MA -> estatistica
    # Constantes
    ## Filtros
    order = 4
    ## Janelamento ERDS
    w_size = 0.5
    noverlap = 0.5
    ref = 1
    ## MA
    w = 3

    vol2 = len(eeg)
    n_ch = np.size(eeg[0], 1)
    n = np.size(eeg[0], 2)

    # ERP
    erp_vol = np.zeros((vol2, n_ch, n))
    for ind in range(vol2):
        erp_vol[ind, :, :] = np.mean(eeg[ind], axis=0)
    erp = np.mean(erp_vol, axis=0)

    # ERDS - Subtração do ERP, Filtragem por Banda
    eeg_filt = []
    for ind in range(vol2):
        eeg_epocas_filt = []
        eeg_epocas_vol = eeg[ind]
        for ind_epoca in range(np.size(eeg_epocas_vol, 0)):
            sinal1 = eeg_epocas_vol[ind_epoca, :, :] - erp[:, :]  # -ERP
            sinal_aux1 = my.my_filter(sinal1, fs, order, freq1, freq2)
            eeg_epocas_filt.append(sinal_aux1)
        eeg_epocas_filt = np.array(eeg_epocas_filt)
        eeg_filt.append(eeg_epocas_filt)

    # ERDS
    ERDS_vol = []
    ERDS_vol_N = []
    for ind in range(vol2):
        ERDS_ind = [] #janela
        ERDS_ind_N = [] #amostra
        eeg_epocas_vol = eeg_filt[ind]
        for ind_ch in range(n_ch):
            eeg_ch = eeg_epocas_vol[:, ind_ch, :-1]
            eeg_pwr = eeg_ch ** 2
            P_0 = np.mean(eeg_pwr, 0)  # Media das épocas
            P_w = strided_app(P_0, int(w_size * fs), int(noverlap * w_size * fs))  # Janelamento
            R_erds = np.mean(P_0[0:int(ref * fs)])  # Referencia
            P_erds = np.mean(P_w, 1)  # Média de cada janela
            ERDS_ch = ((P_erds - R_erds) / R_erds) * 100
            ERDS_ch_N = ((P_0 - R_erds) / R_erds) * 100
            ERDS_ind.append(ERDS_ch)
            ERDS_ind_N.append(ERDS_ch_N)

        ERDS_vol.append(ERDS_ind)
        ERDS_vol_N.append(ERDS_ind_N)

    ERDS_avg = np.mean(np.array(ERDS_vol), axis=0)  # Média dos voluntários
    ERDS_avg_N = np.mean(np.array(ERDS_vol_N), axis=0)

    # ERDS - Filtragem(Moving Average)
    ERDS_MA = []
    ERDS_N_MA = []
    for ind_ch in range(n_ch):
        ERDS_MA.append(np.convolve(ERDS_avg[ind_ch], np.ones(w) / w, 'valid'))

    t_erds = strided_app(np.linspace(t1, t2, np.size(P_0)), int(w_size * fs), int(noverlap * w_size * fs))
    t_w = np.convolve(t_erds[:, 0], np.ones(w) / w, 'valid')

    for ind_ch in range(n_ch):
        ERDS_N_MA.append(np.convolve(ERDS_avg_N[ind_ch], np.ones(w) / w, 'valid'))

    t_N = np.linspace(t1, t2, np.size(P_0))

    return ERDS_MA, t_w, erp, ERDS_vol, t_erds[:, 0], erp_vol, ERDS_avg, ERDS_avg_N, ERDS_vol_N, ERDS_N_MA, t_N


def erds2(eeg, t1, t2, freq1, freq2, fs):
    # Correção -> Não é Grand Average
    # calculo pela variance
    # ERDS vol não tem filtro MA -> estatistica
    # Constantes
    ## Filtros
    order = 4
    ## Janelamento ERDS
    w_size = 0.5
    noverlap = 0.5
    ref = 1
    ## MA
    w = 3

    vol2 = len(eeg)
    n_ch = np.size(eeg[0], 1)
    n = np.size(eeg[0], 2)

    # ERP
    erp_vol = np.zeros((vol2, n_ch, n))
    for ind in range(vol2):
        erp_vol[ind, :, :] = np.mean(eeg[ind], axis=0)
    erp = np.mean(erp_vol, axis=0)

    # ERDS - Subtração do ERP, Filtragem por Banda
    eeg_filt = []
    for ind in range(vol2):
        eeg_epocas_filt = []
        eeg_epocas_vol = eeg[ind]
        for ind_epoca in range(np.size(eeg_epocas_vol, 0)):
            sinal1 = eeg_epocas_vol[ind_epoca, :, :]
            sinal_aux1 = my.my_filter(sinal1, fs, order, freq1, freq2)
            eeg_epocas_filt.append(sinal_aux1)
        eeg_epocas_filt = np.array(eeg_epocas_filt)
        eeg_filt.append(eeg_epocas_filt)

    # ERDS
    ERDS_vol = []
    ERDS_vol_N = []
    for ind in range(vol2):
        ERDS_ind = [] #janela
        ERDS_ind_N = [] #amostra
        eeg_epocas_vol = eeg_filt[ind]
        for ind_ch in range(n_ch):
            eeg_ch = eeg_epocas_vol[:, ind_ch, :-1]
            P_0 = np.var(eeg_ch, axis=0, ddof=1)
            P_w = strided_app(P_0, int(w_size * fs), int(noverlap * w_size * fs))  # Janelamento
            R_erds = np.mean(P_0[0:int(ref * fs)])  # Referencia
            P_erds = np.mean(P_w, 1)  # Média de cada janela
            ERDS_ch = ((P_erds - R_erds) / R_erds) * 100
            ERDS_ch_N = ((P_0 - R_erds) / R_erds) * 100
            ERDS_ind.append(ERDS_ch)
            ERDS_ind_N.append(ERDS_ch_N)

        ERDS_vol.append(ERDS_ind)
        ERDS_vol_N.append(ERDS_ind_N)

    ERDS_avg = np.mean(np.array(ERDS_vol), axis=0)  # Média dos voluntários
    ERDS_avg_N = np.mean(np.array(ERDS_vol_N), axis=0)

    # ERDS - Filtragem(Moving Average)
    ERDS_MA = []
    ERDS_N_MA = []
    for ind_ch in range(n_ch):
        ERDS_MA.append(np.convolve(ERDS_avg[ind_ch], np.ones(w) / w, 'valid'))

    t_erds = strided_app(np.linspace(t1, t2, np.size(P_0)), int(w_size * fs), int(noverlap * w_size * fs))
    t_w = np.convolve(t_erds[:, 0], np.ones(w) / w, 'valid')

    for ind_ch in range(n_ch):
        ERDS_N_MA.append(np.convolve(ERDS_avg_N[ind_ch], np.ones(w) / w, 'valid'))

    t_N = np.linspace(t1, t2, np.size(P_0))

    return ERDS_MA, t_w, erp, ERDS_vol, t_erds[:, 0], erp_vol, ERDS_avg, ERDS_avg_N, ERDS_vol_N, ERDS_N_MA, t_N


def plotersp(erds_1, erds_2, t_erds, y1, title, line_labels, ch_name):
    fig, axes = plt.subplots(nrows=4, ncols=5, sharey=True, sharex=True, figsize=(12, 8))
    axes = axes.flatten()
    ylim = [-y1, y1]
    ch_v = [0, 5, 10, 1, 6, 11, 16, 4, 9, 14, 3, 8, 13, 18, 2, 7, 12]
    ind = 0
    #fig.subplots_adjust(wspace=0.1, hspace=0.5)
    #minor_ticks_top = np.linspace(t1, t2, 2*(abs(t1) + t2) + 1)
    #major_ticks_top = np.linspace(t1, t2, abs(t1) + t2 + 1)
    majorLocator = MultipleLocator(2)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(1)

    for ind_ch in ch_v:
        plt.suptitle(title)
        l1 = axes[ind_ch].plot(t_erds, erds_1[ind])
        l2 = axes[ind_ch].plot(t_erds, erds_2[ind])
        #axes[ind_ch].set_xticks(major_ticks_top)
        #axes[ind_ch].set_xticks(minor_ticks_top, minor=True)
        axes[ind_ch].xaxis.set_major_locator(majorLocator)
        axes[ind_ch].xaxis.set_major_formatter(majorFormatter)
        axes[ind_ch].xaxis.set_minor_locator(minorLocator)
        axes[ind_ch].grid(which="major", alpha=0.6)
        axes[ind_ch].grid(which="minor", alpha=0.3)
        axes[ind_ch].axvline(x=0, linewidth=1, color='r', linestyle="--")
        axes[ind_ch].axvline(x=2.5, linewidth=1, color='g', linestyle="--")
        axes[ind_ch].set_title(ch_name[ind])
        axes[ind_ch].set_ylim(ylim)
        ind += 1

    fig.delaxes(axes[15])
    fig.delaxes(axes[17])
    fig.delaxes(axes[19])

    fig.tight_layout()
    fig.legend([l1, l2],  # The line objects
               labels=line_labels,  # The labels for each line
               loc="upper right",  # Position of legend
               #borderaxespad=0.1,  # Small spacing around legend box
               # title="Legend Title"  # Title for the legend
               )
    plt.show()


def ploterp(erp_1, erp_2, t1, t2, y1, x1, x2, fs, title, line_labels, ch_name):
    t_erp = np.arange(t1, t2, 1 / fs)
    t_erp_atividade = np.where((np.round(t_erp, 2) >= x1) & (np.round(t_erp, 2) <= x2))
    xlim = (x1, x2)
    ylim = (-y1, y1)

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(12, 8), sharey=True, sharex=True,)
    axes = axes.flatten()
    ch_v = [0, 5, 10, 1, 6, 11, 16, 4, 9, 14, 3, 8, 13, 18, 2, 7, 12]  # Lana
    ch_total = np.arange(20)
    ch_diff = [x for x in ch_total if x not in ch_v]

    majorLocator = MultipleLocator(0.5)
    majorFormatter = FormatStrFormatter('%.2f')
    minorLocator = MultipleLocator(0.1)

    ind = 0
    for ind_ch in ch_v:
        plt.suptitle(title)
        l1 = axes[ind_ch].plot(t_erp[t_erp_atividade], np.squeeze(erp_1[ind, t_erp_atividade]))
        l2 = axes[ind_ch].plot(t_erp[t_erp_atividade], np.squeeze(erp_2[ind, t_erp_atividade]))
        axes[ind_ch].xaxis.set_major_locator(majorLocator)
        axes[ind_ch].xaxis.set_major_formatter(majorFormatter)
        axes[ind_ch].xaxis.set_minor_locator(minorLocator)
        axes[ind_ch].axvline(x=0, linewidth=1, color='r', linestyle="--")
        axes[ind_ch].axvline(x=2.5, linewidth=1, color='g', linestyle="--")
        axes[ind_ch].set_title(ch_name[ind])
        axes[ind_ch].set_ylim(ylim)
        axes[ind_ch].set_xlim(xlim)
        axes[ind_ch].grid(which="major", alpha=0.6)
        axes[ind_ch].grid(which="minor", alpha=0.3)
        ind += 1

    for ind in ch_diff:
        fig.delaxes(axes[ind])

    fig.tight_layout()
    fig.legend((l1, l2),  # The line objects
               labels=line_labels,  # The labels for each line
               # borderaxespad=0,    # Small spacing around legend box
               loc='upper right'
               )

    plt.show()


def topographicmap(erds, tmin, tmax, t0, title, ch_name, vmin, vmax):

    t_atividade = np.where((np.round(t0,2) >= tmin) & (np.round(t0,2) <= tmax))

    ch_row = [0, 1, 2, 0, 1,
              2, 3, 0, 1, 2,
              0, 1, 2, 3, 0,
              1, 2]
    ch_col = [0, 0, 0, 1, 1,
              1, 1, 4, 4, 4,
              3, 3, 3, 3, 2,
              2, 2]

    Zvector = np.zeros(len(erds))
    for ind in range(len(erds)):
        Zind = erds[ind, t_atividade]
        # Zvector[ind] = max(Zind.min(), Zind.max(), key=abs)
        Zvector[ind] = np.mean(Zind)

    Zmatrix = np.zeros((4, 5))
    for ind in range(len(Zvector)):
        Zmatrix[ch_row[ind], ch_col[ind]] = Zvector[ind]
    Zmatrix[Zmatrix == 0] = np.nan

    fig, (ax) = plt.subplots(1, 1)
    plt.title(title)
    Z = np.ma.masked_where(np.isnan(Zmatrix), Zmatrix)
    c = ax.pcolormesh(Z, shading='flat', edgecolors='w', linewidths=10, vmin=vmin, vmax=vmax)
    plt.gca().invert_yaxis()

    for ind in range(len(ch_name)):
        if not (Z.mask[ch_row[ind], ch_col[ind]]):
            plt.text(ch_col[ind] + 0.5, ch_row[ind] + 0.5, ch_name[ind],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='w', fontsize='14'
                     )

    fig.colorbar(c)
    plt.axis('off')
    plt.show()


def erdstest(erds,tmin,tmax,t,ch):
    t_atividade = np.where((np.round(t,2) >= tmin) & (np.round(t,2) <= tmax))
    erds = np.array(erds)
    aux_atividade = np.squeeze(erds[:, :, t_atividade])

    erds_mean =[]
    erds_max = []
    for ind in range(len(ch)):
        aux = np.squeeze(aux_atividade[:, ch[ind], :])
        erds_mean.append(np.mean(aux,1))
        erds_max_vol = []
        for ind2 in range(len(aux_atividade)):
            erds_max_vol.append(max(aux[ind2, :].min(), aux[ind2, :].max(), key=abs))
        erds_max.append(np.array(erds_max_vol))
    return np.array(erds_mean), np.array(erds_max)


def analisestest(erds_mean, voluntarios, ch_name, ch_box, title0):
    fig, ax = plt.subplots(1, figsize=(10,6))
    fig.suptitle('ERDS Mean - ' + title0 )
    ax.boxplot(np.transpose(erds_mean[:, voluntarios]), labels=[ch_name[ind] for ind in ch_box])
    ax.set_ylabel('Amplitude [uV]')

    for ind in range(3):
        ch = [ind*2,ind*2+1]
        a = stats.wilcoxon(erds_mean[ch[0],voluntarios],erds_mean[ch[1],voluntarios], alternative='two-sided')
        x1, x2 = ch[0]+1, ch[1]+1
        #y, h, col = np.max([erds_mean[ch[0],voluntarios], erds_mean[ch[1],voluntarios]]) + 2, 2, 'k'
        y, h, col = np.max([erds_mean[ch[0], voluntarios], erds_mean[ch[1], voluntarios]]) + 3, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, "{:.4f}".format(a[1]), ha='center', va='bottom', color=col)

    plt.show()


def outlier(arr):
    q1 = np.quantile(arr, 0.25)
    q3 = np.quantile(arr, 0.75)
    iqr = q3 - q1

    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    outliers = arr[(arr <= lower_bound) | (arr >= upper_bound)]
    print('outliers')
    print(outliers)

    vol_ind = []
    for ind in range(len(outliers)):
        vol_ind.append(np.where(arr==outliers[ind]))
    vol_ind = np.squeeze(np.array(vol_ind))
    print('Voluntário')
    print(vol_ind)
    return vol_ind


def analisesplot(erds_mean, ch, ch_name, ch_box, title0):

    erds_teste = np.transpose(erds_mean[ch,:])
    aux_teste = erds_mean[ch[0]]-erds_mean[ch[1]]
    MV_testes = np.column_stack((erds_teste,aux_teste))
    label_ch = [ch_name[ind] for ind in ch_box[ch]]
    label_ch.append(ch_name[ch_box[ch[0]]] + ' - ' + ch_name[ch_box[ch[1]]])

    fig, ax = plt.subplots(2, figsize=(10,6))
    fig.suptitle('ERDS Mean - ' + title0)
    ax[0].boxplot(MV_testes, labels=label_ch)
    ax[0].set_ylabel('Amplitude [uV]')
    ax[1].plot(erds_mean[ch[0]],'xr', label=ch_name[ch_box[ch[0]]])
    ax[1].plot(erds_mean[ch[1]],'xb', label=ch_name[ch_box[ch[1]]])
    ax[1].set_ylabel(ch_name[ch_box[ch[0]]] + ' - ' + ch_name[ch_box[ch[1]]])
    ax[1].legend()
    plt.show()


def stderdsplot(erds_ch, t_vol, voluntarios, title):
    mean_ch = np.mean(erds_ch, axis=0)
    mean_aux = np.mean(erds_ch)
    # std_aux = np.std(mean_ch)
    std_aux = np.std(erds_ch)

    ylim = [-100, 100]
    majorLocator = MultipleLocator(2)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(1)

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.plot(t_vol, np.transpose(erds_ch))
    ax.legend(voluntarios, loc='center right', bbox_to_anchor=(-0.05, 0.5))

    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.grid(which="major", alpha=0.6)
    ax.grid(which="minor", alpha=0.3)
    texto = ax.get_yticklabels()[0]
    ax.yaxis.tick_right()
    ax.axvline(x=0, linewidth=1, color='r', linestyle="--")
    ax.axvline(x=2.5, linewidth=1, color='g', linestyle="--")
    ax.axhline(y=std_aux, linewidth=1.5, color='k', linestyle="--")
    ax.axhline(y=std_aux, linewidth=1.5, color='k', linestyle="--")
    ax.axhline(y=-std_aux, linewidth=1.5, color='k', linestyle="--")
    ax.axhline(y=2 * std_aux, linewidth=1.5, color='k', linestyle="--")
    ax.axhline(y=-2 * std_aux, linewidth=1.5, color='k', linestyle="--")
    ax.axhline(y=3 * std_aux, linewidth=1.5, color='k', linestyle="--")
    ax.axhline(y=-3 * std_aux, linewidth=1.5, color='k', linestyle="--")
    ax.set_title(title)
    trans = transforms.blended_transform_factory(
        texto.get_transform(), ax.transData)
    ax.text(0, std_aux, 'std', color="red", transform=trans,
            ha="right", va="center")
    ax.text(0, 2 * std_aux, '2*std', color="red", transform=trans,
            ha="right", va="center")
    ax.text(0, 3 * std_aux, '3*std', color="red", transform=trans,
            ha="right", va="center")
    ax.set_ylim(ylim)
    plt.show()

def plotersp2(erds_1, erds_2, t_erds, y1, title, line_labels, ch_name):
    fig, axes = plt.subplots(nrows=4, ncols=3, sharey=True, sharex=True, figsize=(12, 8))
    axes = axes.flatten()
    ylim = [-y1, y1]
    #ch_v = [0, 5, 10, 1, 6, 11, 16, 4, 9, 14, 3, 8, 13, 18, 2, 7, 12]
    ch_v = [3,14,10,4,15,11,5,16,12,6,13,13]
    ind = 0

    majorLocator_x = MultipleLocator(2)
    majorFormatter_x = FormatStrFormatter('%d')
    minorLocator_x = MultipleLocator(1)
    majorLocator_y = MultipleLocator(25)
    majorFormatter_y = FormatStrFormatter('%d')

    for ind_ch in ch_v:
        plt.suptitle(title)
        l1 = axes[ind].plot(t_erds, erds_1[ind_ch], color='b')
        l2 = axes[ind].plot(t_erds, erds_2[ind_ch], color='r')
        #axes[ind_ch].set_xticks(major_ticks_top)
        #axes[ind_ch].set_xticks(minor_ticks_top, minor=True)
        axes[ind].xaxis.set_major_locator(majorLocator_x)
        axes[ind].xaxis.set_major_formatter(majorFormatter_x)
        axes[ind].xaxis.set_minor_locator(minorLocator_x)
        axes[ind].yaxis.set_major_locator(majorLocator_y)
        axes[ind].yaxis.set_major_formatter(majorFormatter_y)
        axes[ind].grid(which="major", alpha=0.6)
        axes[ind].grid(which="minor", alpha=0.3)
        axes[ind].axvline(x=0, linewidth=1, color='r', linestyle="--")
        axes[ind].axvline(x=2.5, linewidth=1, color='g', linestyle="--")
        axes[ind].set_title(ch_name[ind_ch])
        axes[ind].set_ylim(ylim)
        ind += 1

    fig.delaxes(axes[10])
    axes[9].set_xlabel("Time (s)")
    axes[9].set_ylabel('% ERD/ERS')
    axes[9].yaxis.set_label_coords(-0.09, 0.5)
    fig.legend([l1, l2],  # The line objects
               labels=line_labels,  # The labels for each line
               loc="upper right",  # Position of legend
               #borderaxespad=0.1,  # Small spacing around legend box
               # title="Legend Title"  # Title for the legend
               )
    fig.tight_layout()
    plt.show()

def ploterp2(erp_1, erp_2, t1, t2, y1, x1, x2, fs, title, line_labels, ch_name):
    t_erp = np.arange(t1, t2, 1 / fs)
    t_erp_atividade = np.where((np.round(t_erp, 2) >= x1) & (np.round(t_erp, 2) <= x2))
    xlim = (x1, x2)
    ylim = (-y1, y1)

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 8), sharey=True, sharex=True,)
    axes = axes.flatten()
    ch_v = [3,14,10,4,15,11,5,16,12,6,13,13]

    majorLocator = MultipleLocator(0.5)
    majorFormatter = FormatStrFormatter('%.1f')
    minorLocator = MultipleLocator(0.1)
    scale_y = 1e-6

    ind = 0
    for ind_ch in ch_v:
        plt.suptitle(title)
        l1 = axes[ind].plot(t_erp[t_erp_atividade], np.squeeze(erp_1[ind_ch, t_erp_atividade]), color='b')
        l2 = axes[ind].plot(t_erp[t_erp_atividade], np.squeeze(erp_2[ind_ch, t_erp_atividade]), color='r')
        axes[ind].xaxis.set_major_locator(majorLocator)
        axes[ind].xaxis.set_major_formatter(majorFormatter)
        axes[ind].xaxis.set_minor_locator(minorLocator)
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))
        axes[ind].yaxis.set_major_formatter(ticks_y)
        axes[ind].axvline(x=0, linewidth=1, color='r', linestyle="--")
        axes[ind].axvline(x=2.5, linewidth=1, color='g', linestyle="--")
        axes[ind].set_title(ch_name[ind_ch])
        axes[ind].set_ylim(ylim)
        axes[ind].set_xlim(xlim)
        axes[ind].grid(which="major", alpha=0.6)
        axes[ind].grid(which="minor", alpha=0.3)
        ind += 1

    axes[9].set_xlabel("Time (s)")
    axes[9].set_ylabel('Amplitude [uV]')
    axes[11].set_xlabel("Time (s)")
    fig.delaxes(axes[10])

    fig.legend((l1, l2),  # The line objects
               labels=line_labels,  # The labels for each line
               # borderaxespad=0,    # Small spacing around legend box
               loc='upper right'
               )
    fig.tight_layout()
    plt.show()

def topographicmap2(erds, tmin, tmax, t0, title, ch_name, vmin, vmax):

    t_atividade = np.where((np.round(t0,2) >= tmin) & (np.round(t0,2) <= tmax))

    ch_v = [3, 14, 10,
            4, 15, 11,
            5, 16, 12,
            6, 13]

    ch_row = [0, 0, 0,
              1, 1, 1,
              2, 2, 2,
              3, 3]

    ch_col = [0, 1, 2,
              0, 1, 2,
              0, 1, 2,
              0, 2]

    Zvector = np.zeros(len(ch_v))
    ind = 0
    for ind_ch in ch_v:
        Zind = erds[ind_ch, t_atividade]
        # Zvector[ind] = max(Zind.min(), Zind.max(), key=abs)
        Zvector[ind] = np.mean(Zind)
        ind += 1

    Zmatrix = np.zeros((4, 3))
    for ind in range(len(Zvector)):
        Zmatrix[ch_row[ind], ch_col[ind]] = Zvector[ind]
    Zmatrix[Zmatrix == 0] = np.nan

    fig, (ax) = plt.subplots(1, 1)
    plt.title(title)
    Z = np.ma.masked_where(np.isnan(Zmatrix), Zmatrix)
    c = ax.pcolormesh(Z, shading='flat', edgecolors='w', linewidths=10, vmin=vmin, vmax=vmax)
    plt.gca().invert_yaxis()

    for ind in range(len(ch_v)):
        if not (Z.mask[ch_row[ind], ch_col[ind]]):
            plt.text(ch_col[ind] + 0.5, ch_row[ind] + 0.5, ch_name[ch_v[ind]],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='w', fontsize='14'
                     )

    fig.colorbar(c)
    plt.axis('off')
    plt.show()


def erdsvol(eeg, t1, t2, freq1, freq2, fs):
    # Mudança no ERDS_vol para plot (plot precisa do filtro MA)
    # Para estatistica não precisa do MA

    # Constantes
    ## Filtros
    order = 4
    ## Janelamento ERDS
    w_size = 0.5
    noverlap = 0.5
    ref = 1
    ## MA
    w = 3

    vol2 = len(eeg)
    n_ch = np.size(eeg[0], 1)
    n = np.size(eeg[0], 2)

    # ERP
    erp_vol = np.zeros((vol2, n_ch, n))
    for ind in range(vol2):
        erp_vol[ind, :, :] = np.mean(eeg[ind], axis=0)
    erp = np.mean(erp_vol, axis=0)

    # ERDS - Subtração do ERP, Filtragem por Banda
    eeg_filt = []
    for ind in range(vol2):
        eeg_epocas_filt = []
        eeg_epocas_vol = eeg[ind]
        for ind_epoca in range(np.size(eeg_epocas_vol, 0)):
            sinal1 = eeg_epocas_vol[ind_epoca, :, :] - erp[:, :]  # -ERP
            sinal_aux1 = my.my_filter(sinal1, fs, order, freq1, freq2)
            eeg_epocas_filt.append(sinal_aux1)
        eeg_epocas_filt = np.array(eeg_epocas_filt)
        eeg_filt.append(eeg_epocas_filt)

    # ERDS
    ERDS_vol = []
    ERDS_vol_N = []
    for ind in range(vol2):
        ERDS_ind = []  # janela
        ERDS_ind_N = []  # amostra
        eeg_epocas_vol = eeg_filt[ind]
        for ind_ch in range(n_ch):
            eeg_ch = eeg_epocas_vol[:, ind_ch, :-1]
            eeg_pwr = eeg_ch ** 2
            P_0 = np.mean(eeg_pwr, 0)  # Media das épocas
            P_w = strided_app(P_0, int(w_size * fs), int(noverlap * w_size * fs))  # Janelamento
            R_erds = np.mean(P_0[0:int(ref * fs)])  # Referencia
            P_erds = np.mean(P_w, 1)  # Média de cada janela
            ERDS_ch = ((P_erds - R_erds) / R_erds) * 100
            ERDS_ch_N = ((P_0 - R_erds) / R_erds) * 100
            ERDS_ind.append(ERDS_ch)
            ERDS_ind_N.append(ERDS_ch_N)

        ERDS_vol.append(ERDS_ind)
        ERDS_vol_N.append(ERDS_ind_N)

    ERDS_avg = np.mean(np.array(ERDS_vol), axis=0)  # Média dos voluntários

    # ERDS - Filtragem(Moving Average)
    ERDS_MA = []

    for ind_ch in range(n_ch):
        ERDS_MA.append(np.convolve(ERDS_avg[ind_ch], np.ones(w) / w, 'same'))

    ERDS_vol_MA = []

    for ind_v in range(vol2):
        ERDS_vol_MA_ch = []
        ERDS_vol_aux = ERDS_vol[ind_v]
        for ind_ch in range(n_ch):

            ERDS_vol_MA_ch.append(np.convolve(ERDS_vol_aux[ind_ch], np.ones(w) / w, 'same'))

        ERDS_vol_MA.append(ERDS_vol_MA_ch)
    t_erds = strided_app(np.linspace(t1, t2, np.size(P_0)), int(w_size * fs), int(noverlap * w_size * fs))
    # t_w = np.convolve(t_erds[:, 0], np.ones(w) / w, 'valid')

    return ERDS_MA, erp, ERDS_vol, t_erds[:, 0], erp_vol, ERDS_vol_MA


def erdsvol2(eeg, t1, t2, freq1, freq2, fs):
    # Mudança no ERDS_vol para plot (plot precisa do filtro MA)
    # Para estatistica não precisa do MA

    # Constantes
    ## Filtros
    order = 4
    ## Janelamento ERDS
    w_size = 0.5
    noverlap = 0.5
    ref = 1
    ## MA
    w = 3

    vol2 = len(eeg)
    n_ch = np.size(eeg[0], 1)
    n = np.size(eeg[0], 2)

    # ERP
    erp_vol = np.zeros((vol2, n_ch, n))
    for ind in range(vol2):
        erp_vol[ind, :, :] = np.mean(eeg[ind], axis=0)
    erp = np.mean(erp_vol, axis=0)

    # ERDS - Subtração do ERP, Filtragem por Banda
    eeg_filt = []
    for ind in range(vol2):
        eeg_epocas_filt = []
        eeg_epocas_vol = eeg[ind]
        for ind_epoca in range(np.size(eeg_epocas_vol, 0)):
            sinal1 = eeg_epocas_vol[ind_epoca, :, :]
            sinal_aux1 = my.my_filter(sinal1, fs, order, freq1, freq2)
            eeg_epocas_filt.append(sinal_aux1)
        eeg_epocas_filt = np.array(eeg_epocas_filt)
        eeg_filt.append(eeg_epocas_filt)

    # ERDS
    ERDS_vol = []
    ERDS_vol_N = []
    for ind in range(vol2):
        ERDS_ind = []  # janela
        ERDS_ind_N = []  # amostra
        eeg_epocas_vol = eeg_filt[ind]
        for ind_ch in range(n_ch):
            eeg_ch = eeg_epocas_vol[:, ind_ch, :-1]
            P_0 = np.var(eeg_ch, axis=0, ddof=1)
            P_w = strided_app(P_0, int(w_size * fs), int(noverlap * w_size * fs))  # Janelamento
            R_erds = np.mean(P_0[0:int(ref * fs)])  # Referencia
            P_erds = np.mean(P_w, 1)  # Média de cada janela
            ERDS_ch = ((P_erds - R_erds) / R_erds) * 100
            ERDS_ch_N = ((P_0 - R_erds) / R_erds) * 100
            ERDS_ind.append(ERDS_ch)
            ERDS_ind_N.append(ERDS_ch_N)

        ERDS_vol.append(ERDS_ind)
        ERDS_vol_N.append(ERDS_ind_N)

    ERDS_avg = np.mean(np.array(ERDS_vol), axis=0)  # Média dos voluntários

    # ERDS - Filtragem(Moving Average)
    ERDS_MA = []

    for ind_ch in range(n_ch):
        ERDS_MA.append(np.convolve(ERDS_avg[ind_ch], np.ones(w) / w, 'same'))

    ERDS_vol_MA = []

    for ind_v in range(vol2):
        ERDS_vol_MA_ch = []
        ERDS_vol_aux = ERDS_vol[ind_v]
        for ind_ch in range(n_ch):

            ERDS_vol_MA_ch.append(np.convolve(ERDS_vol_aux[ind_ch], np.ones(w) / w, 'same'))

        ERDS_vol_MA.append(ERDS_vol_MA_ch)
    t_erds = strided_app(np.linspace(t1, t2, np.size(P_0)), int(w_size * fs), int(noverlap * w_size * fs))
    # t_w = np.convolve(t_erds[:, 0], np.ones(w) / w, 'valid')

    return ERDS_MA, erp, ERDS_vol, t_erds[:, 0], erp_vol, ERDS_vol_MA


def ploterspanalise(erds, erds_vol, t_erds, y1, title, ch_name, ch_box, c):
    tini = -2
    tfim = 6
    janela = 0.5
    tmin_v = np.arange(tini, tfim, janela)
    tmax_v = np.arange(tini + janela, tfim + janela, janela)
    # alpha = 0.05
    # tini_1 = -2
    # tfim_1 = 4.5
    # tini_2 = 5
    # tfim_2 = 6
    # janela = 0.5
    # tmin_1 = np.arange(tini_1, tfim_1, janela)
    # tmax_1 = np.arange(tini_1 + janela, tfim_1 + janela, janela)
    # tmin_2 = np.arange(tini_2, tfim_2, janela)
    # tmax_2 = np.arange(tini_2 + janela, tfim_2 + janela, janela)
    # tmin_v = np.concatenate((tmin_1, tmin_2), axis=0)
    # tmax_v = np.concatenate((tmax_1, tmax_2), axis=0)

    alpha = 0.05

    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True, figsize=(6, 8))
    plt.suptitle(title)
    axes = axes.flatten()
    ylim = [-y1, y1]

    majorLocator_x = MultipleLocator(1)
    majorFormatter_x = FormatStrFormatter('%d')
    minorLocator_x = MultipleLocator(0.5)
    majorLocator_y = MultipleLocator(25)
    majorFormatter_y = FormatStrFormatter('%d')

    ind_plot = 0
    for ind_ch in range(int(len(ch_box) / 2)):
        l1 = axes[ind_plot].plot(t_erds, erds[ch_box[ind_ch * 2]], label=ch_name[ch_box[ind_ch * 2]])
        l2 = axes[ind_plot].plot(t_erds, erds[ch_box[ind_ch * 2 + 1]], label=ch_name[ch_box[ind_ch * 2 + 1]])
        axes[ind_plot].xaxis.set_major_locator(majorLocator_x)
        axes[ind_plot].xaxis.set_major_formatter(majorFormatter_x)
        axes[ind_plot].xaxis.set_minor_locator(minorLocator_x)
        axes[ind_plot].yaxis.set_major_locator(majorLocator_y)
        axes[ind_plot].yaxis.set_major_formatter(majorFormatter_y)
        axes[ind_plot].grid(which="major", alpha=0.6)
        axes[ind_plot].grid(which="minor", alpha=0.3)
        axes[ind_plot].axvline(x=0, linewidth=1, color='r', linestyle="--")
        axes[ind_plot].axvline(x=2.5, linewidth=1, color='g', linestyle="--")
        name = ch_name[ch_box[ind_ch * 2]] + ' - ' + ch_name[ch_box[ind_ch * 2 + 1]]
        axes[ind_plot].set_title(name)
        axes[ind_plot].set_ylim(ylim)
        axes[ind_plot].legend(loc="upper left")
        axes[ind_plot].set_ylabel('% ERD/ERS')
        ind_plot += 1

    col_t = []

    for ind_t in range(len(tmin_v)):
        erds_mean, _ = erdstest(erds_vol, tmin_v[ind_t], tmax_v[ind_t], t_erds, ch_box)
        row_ch = []
        for ind in range(int(len(ch_box) / 2)):
            ch = [ind * 2, ind * 2 + 1]
            a = stats.wilcoxon(erds_mean[ch[0], :], erds_mean[ch[1], :], alternative='two-sided')
            row_ch.append(a[1])
            if a[1] <= alpha:
                axes[ind].axvspan(tmin_v[ind_t], tmax_v[ind_t], color=c)

        col_t.append(row_ch)

    table = np.array(col_t).transpose()

    fig.tight_layout()
    plt.show()

    return table, tmin_v


def ploterspMVIMG(MV_erds, MV_erds_vol, IM_erds, IM_erds_vol, t_erds, y1, title, ch_name, ch_box, c, label):

    tini = -2
    tfim = 6
    janela = 0.5
    tmin_v = np.arange(tini, tfim, janela)
    tmax_v = np.arange(tini + janela, tfim + janela, janela)
    alpha = 0.05
    # tini_1 = -2
    # tfim_1 = 4.5
    # tini_2 = 5
    # tfim_2 = 6
    # janela = 0.5
    # tmin_1 = np.arange(tini_1, tfim_1, janela)
    # tmax_1 = np.arange(tini_1 + janela, tfim_1 + janela, janela)
    # tmin_2 = np.arange(tini_2, tfim_2, janela)
    # tmax_2 = np.arange(tini_2 + janela, tfim_2 + janela, janela)
    # tmin_v = np.concatenate((tmin_1, tmin_2), axis=0)
    # tmax_v = np.concatenate((tmax_1, tmax_2), axis=0)

    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True, figsize=(6, 8))
    plt.suptitle(title)
    axes = axes.flatten()
    ylim = [-y1, y1]

    majorLocator_x = MultipleLocator(1)
    majorFormatter_x = FormatStrFormatter('%d')
    minorLocator_x = MultipleLocator(0.5)
    majorLocator_y = MultipleLocator(25)
    majorFormatter_y = FormatStrFormatter('%d')

    ind_plot = 0
    for ind_ch in range(int(len(ch_box))):
        l1 = axes[ind_plot].plot(t_erds, MV_erds[ch_box[ind_ch]], label=label[0])
        l2 = axes[ind_plot].plot(t_erds, IM_erds[ch_box[ind_ch]], label=label[1])
        axes[ind_plot].xaxis.set_major_locator(majorLocator_x)
        axes[ind_plot].xaxis.set_major_formatter(majorFormatter_x)
        axes[ind_plot].xaxis.set_minor_locator(minorLocator_x)
        axes[ind_plot].yaxis.set_major_locator(majorLocator_y)
        axes[ind_plot].yaxis.set_major_formatter(majorFormatter_y)
        axes[ind_plot].grid(which="major", alpha=0.6)
        axes[ind_plot].grid(which="minor", alpha=0.3)
        axes[ind_plot].axvline(x=0, linewidth=1, color='r', linestyle="--")
        axes[ind_plot].axvline(x=2.5, linewidth=1, color='g', linestyle="--")
        axes[ind_plot].set_title(ch_name[ch_box[ind_ch]])
        axes[ind_plot].set_ylim(ylim)
        axes[ind_plot].legend(loc="upper left")
        axes[ind_plot].set_ylabel('% ERD/ERS')
        ind_plot += 1

    col_t = []

    for ind_t in range(len(tmin_v)):
        MV_erds_mean, _ = erdstest(MV_erds_vol, tmin_v[ind_t], tmax_v[ind_t], t_erds, ch_box)
        IM_erds_mean, _ = erdstest(IM_erds_vol, tmin_v[ind_t], tmax_v[ind_t], t_erds, ch_box)
        row_ch = []
        for ind in range(int(len(ch_box))):
            a = stats.wilcoxon(MV_erds_mean[ind, :], IM_erds_mean[ind, :], alternative='two-sided')
            row_ch.append(a[1])
            if a[1] <= alpha:
                axes[ind].axvspan(tmin_v[ind_t], tmax_v[ind_t], color=c)

        col_t.append(row_ch)

    table = np.array(col_t).transpose()

    fig.tight_layout()
    plt.show()

    return table, tmin_v

def erpcaracteristica(erp, t, t_inicial, t_final, canais):
    deltat = (t>t_inicial)&(t<t_final)
    erp_canais = erp[canais,:-1]
    erp_deltat = erp_canais[:,deltat] #janela de tempo

    # Max
    erp_max = np.amax(erp_deltat,axis=1)
    max_ind = []
    for i in range(len(canais)):
        max_aux = np.where(erp_canais[i,:]==erp_max[i])[0] #procurando no erp o máximo da janela
        max_ind.append(max_aux[0])
    max_ind = np.array(max_ind)

    # Min
    min_ind = []
    erp_min = []
    t_final_min = t_final + 0.4
    for i in range(len(canais)):
        deltat_min = (t > t[max_ind[i]]) & (t < t_final_min)
        erp_deltamin = erp_canais[i, deltat_min]
        erp_aux_min = np.amin(erp_deltamin)
        min_aux = np.where(erp_canais[i, :]==erp_aux_min)[0] #procurando no erp o mínimo da janela
        min_ind.append(min_aux[0])
        erp_min.append(erp_aux_min)
    erp_min = np.array(erp_min)
    min_ind = np.array(min_ind)
    return erp_max, max_ind, erp_min, min_ind

def ploterpanalise(erp,erp_max, max_ind, erp_min, min_ind, t, y1, x1, x2, canais, ch_name,title):
    t_erp_atividade = np.where((np.round(t, 2) >= x1) & (np.round(t, 2) <= x2))
    xlim = (x1, x2)
    ylim = (-y1, y1)

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 8), sharey=True, sharex=True)
    plt.suptitle(title)
    axes = axes.flatten()
    majorLocator = MultipleLocator(0.5)
    majorFormatter = FormatStrFormatter('%.1f')
    minorLocator = MultipleLocator(0.1)
    scale_y = 1e-6
    ind_ax = 0
    ind = 0
    for ind_ch in canais:
        if ind_ax == 10:
            ind_ax +=1
        l1 = axes[ind_ax].plot(t[t_erp_atividade], np.squeeze(erp[ind_ch, t_erp_atividade]), color='b')
        axes[ind_ax].xaxis.set_major_locator(majorLocator)
        axes[ind_ax].xaxis.set_major_formatter(majorFormatter)
        axes[ind_ax].xaxis.set_minor_locator(minorLocator)
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))
        axes[ind_ax].yaxis.set_major_formatter(ticks_y)
        axes[ind_ax].axvline(x=0, linewidth=1, color='r', linestyle="--")
        axes[ind_ax].axvline(x=2.5, linewidth=1, color='g', linestyle="--")
        axes[ind_ax].set_title(ch_name[ind_ch])
        axes[ind_ax].set_ylim(ylim)
        axes[ind_ax].set_xlim(xlim)
        axes[ind_ax].grid(which="major", alpha=0.6)
        axes[ind_ax].grid(which="minor", alpha=0.3)
        axes[ind_ax].plot(t[max_ind[ind]], erp_max[ind], "o", color='r')
        axes[ind_ax].plot(t[min_ind[ind]], erp_min[ind], "x", color='r')
        ind_ax += 1
        ind += 1
    axes[9].set_xlabel("Time (s)")
    axes[9].set_ylabel('Amplitude [uV]')
    axes[11].set_xlabel("Time (s)")
    fig.delaxes(axes[10])

    fig.tight_layout()
    plt.show()

def erpestatistica(erp, max_ind, min_ind, tamanho_janela, fs, canais):
    erp_vol = erp[:,canais,:-1]
    max_ch_mean = []
    for canal in range(len(canais)):
        max_vol_mean = []
        for voluntario in range(len(erp_vol)):
            max_vol = erp_vol[voluntario,:,:]
            max_vol = max_vol[canal,(max_ind[canal]-int(tamanho_janela*fs/1000)):(max_ind[canal]+int(tamanho_janela*fs/1000))]
            max_vol_mean.append(np.mean(max_vol))
        max_ch_mean.append(max_vol_mean)

    min_ch_mean = []
    for canal in range(len(canais)):
        min_vol_mean = []
        for voluntario in range(len(erp_vol)):
            min_vol = erp_vol[voluntario,:,:]
            min_vol = min_vol[canal,(min_ind[canal]-int(tamanho_janela*fs/1000)):(min_ind[canal]+int(tamanho_janela*fs/1000))]
            min_vol_mean.append(np.mean(min_vol))
        min_ch_mean.append(min_vol_mean)

    return max_ch_mean, min_ch_mean

def boxerp(caracteristica1,caracteristica2, ch_name, canais, title, legend):
    aux = 2 * np.arange(len(canais))
    dist = 0.3
    fig, ax = plt.subplots(1, figsize=(10, 6))
    fig.suptitle(title)
    c1 = 'red'
    c2 = 'blue'
    scale_y = 1e-6
    bp1 = ax.boxplot(caracteristica1, positions=aux - dist, widths=0.4, patch_artist=True,
                     boxprops=dict(facecolor='None', color=c1),
                     capprops=dict(color=c1),
                     whiskerprops=dict(color=c1),
                     flierprops=dict(color=c1, markeredgecolor=c1),
                     medianprops=dict(color=c1), )

    bp2 = ax.boxplot(caracteristica2, positions=aux + dist, widths=0.4, patch_artist=True,
                     boxprops=dict(facecolor='None', color=c2),
                     capprops=dict(color=c2),
                     whiskerprops=dict(color=c2),
                     flierprops=dict(color=c2, markeredgecolor=c2),
                     medianprops=dict(color=c2), )

    for ind in range(len(canais)):
        a = stats.wilcoxon(caracteristica1[ind], caracteristica2[ind], alternative='two-sided')
        y, h, col = np.max([np.max(caracteristica1[ind]), np.max(caracteristica2[ind])]) + 3 * scale_y, 2 * scale_y, 'k'
        ax.plot([aux[ind] - dist, aux[ind] - dist, aux[ind] + dist, aux[ind] + dist], [y, y + h, y + h, y], lw=1.5,
                c=col)
        ax.text((aux[ind]), y + h, "{:.4f}".format(a[1]), ha='center', va='bottom', color=col)

    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))
    ax.yaxis.set_major_formatter(ticks_y)
    ax.set_ylabel('Amplitude [uV]')
    ax.set_xticks(aux)
    ax.set_xticklabels([ch_name[ind] for ind in canais])
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], legend)

    plt.show()