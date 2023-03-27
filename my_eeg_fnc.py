import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
from matplotlib.ticker import (AutoMinorLocator)

'''
My functions to work with EEG
5/19/2021
    -filter
    -psd
    -plot
    -montage
'''


def my_montage(raw, ch_name):
    # Montage channel choice
    montage = mne.channels.make_standard_montage('standard_1020')
    ind = [i for (i, channel) in enumerate(montage.ch_names) if channel in ch_name]
    montage_new = montage.copy()
    # Keep only the desired channels
    montage_new.ch_names = [montage.ch_names[x] for x in ind]
    kept_channel_info = [montage.dig[x + 3] for x in ind]
    # Keep the first three rows as they are the points information
    montage_new.dig = montage.dig[0:3] + kept_channel_info
    raw.set_montage(montage_new)
    return raw


def my_filter(x, fs, order, freq1, freq2):
    w1 = freq1 / (fs / 2)  # Normalize the frequency
    [b1, a1] = signal.butter(order, w1, btype='high')
    x_aux = signal.filtfilt(b1, a1, x)
    w2 = freq2 / (fs / 2)
    [b2, a2] = signal.butter(order, w2, btype='low')
    x_out = signal.filtfilt(b2, a2, x_aux)
    return x_out


def my_notch(x, fs, freq, q):
    b1, a1 = signal.iirnotch(freq, q, fs)
    x_out = signal.lfilter(b1, a1, x)
    #f1n = freq-1
    #f2n = freq+1
    #w1 = f1n / (fs/2)  # Normalize the frequency
    #w2 = f2n / (fs/2)
    #b1, a1 = signal.butter(order, [w1, w2], btype='bandstop')
    #x_out = signal.filtfilt(b1, a1, x)

    return x_out


def my_plot_psd(x, fs, title):
    x = x / 1e-6
    plt.figure(figsize=(10, 4))
    f, Pxx_den = signal.welch(x, fs, nperseg=2048)
    plt.plot(f, Pxx_den.mean(axis=0))
    plt.yscale('log')
    plt.xlim((0, 280))
    plt.ylim((10e-12, 10e2))
    plt.grid(which='both', axis='both')
    plt.ylabel('PSD [V²/Hz] (dB)')
    plt.xlabel('Frequency [Hz]')
    plt.title(title)
    plt.show()


def my_plot_fft(x, fs, title, ch_name):
    x = x.transpose()
    n = np.size(x, 0)
    dt = 1 / fs
    yf = fft(x, axis=0)
    yf_plot = 2.0 / n * np.abs(yf[0:n // 2])
    xfp = np.linspace(0.0, 1.0 / (2.0 * dt), n // 2)
    plt.figure(figsize=(10, 4))
    plt.plot(xfp, yf_plot, label=ch_name)
    plt.xlim((-1, 300))
    plt.legend(loc='best')
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Single-Sided Amplitude Spectrum |X|')
    plt.grid()
    plt.show()


def my_eeg_eog(x, eog, event, fs, canal, title):
    x = x/1e-6
    eog = eog/1e-6
    t = np.arange(0, len(x) / fs, 1 / fs)
    pos = t[np.where(event > 0)]
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(211)
    plt.plot(t, x)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.grid()
    plt.ylim((-200, 200))
    plt.ylabel(canal+' [uV]')
    plt.title(title)
    plt.xlim((0, 20))
    for _x in pos:
        plt.axvline(_x, linewidth=1, color='r')
        plt.axvline(_x+4, linewidth=1, color='r', linestyle="--")

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(t, eog)
    plt.grid()
    plt.ylim((-200, 200))
    plt.ylabel('EoG'+ ' [uV]')
    plt.xlabel('Time [s]')
    for _x in pos:
        plt.axvline(_x, linewidth=1, color='r')
        plt.axvline(_x+4, linewidth=1, color='r', linestyle="--")

    plt.show()


def my_time_comparison(x, x_f, fs, canal, ylim=100):
    x = x / 1e-6
    x_f = x_f / 1e-6
    t = np.arange(0, len(x) / fs, 1 / fs)
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(211)
    plt.plot(t, x)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.grid()
    plt.ylim((-ylim, ylim))
    plt.ylabel('Não Filtrado')
    #plt.ylabel('Filtrado')
    plt.title(canal)
    plt.xlim((0, 10))
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.plot(t, x_f)
    plt.grid()
    plt.ylabel('[uV]')
    plt.ylabel('Filtrado')
    #plt.ylabel('Filtrado + Notch')
    plt.xlabel('Time [s]')
    plt.show()


def my_eeg_plot(x, fs, ch_name):
    x = x / 1e-6
    t = np.arange(0, len(x) / fs, 1 / fs)
    plt.figure(figsize=(10, 8))
    a = np.size(x, 0)
    b = 1
    c = 1
    for ind in ch_name:
        plt.subplot(a, b, c)
        plt.plot(t, x)
        plt.grid()
        plt.ylim((-100, 100))
        plt.ylabel(ch_name[ind])
        plt.xlim((0, 10))
        c += 1
    plt.show()




def my_fft_comparison(x, x_f, fs):
    x = x.transpose()
    x_f = x_f.transpose()
    n = np.size(x, 0)
    dt = 1 / fs
    xfp = np.linspace(0.0, 1.0 / (2.0 * dt), n // 2)
    yf = fft(x, axis=0)
    yf_plot = 2.0 / n * np.abs(yf[0:n // 2])
    yf_f = fft(x_f, axis=0)
    yf_f_plot = 2.0 / n * np.abs(yf_f[0:n // 2])

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(211)
    plt.title('FFT')
    plt.plot(xfp, yf_plot)
    plt.ylabel('Não Filtrado [V]')
    plt.ylim((0, 1e-5))
    plt.xlim((-1, 300))
    plt.grid()
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.plot(xfp, yf_f_plot)
    plt.ylabel('Filtrado [V]')
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    plt.show()


def my_psd_comparison(x, x_f, fs, cut):
    x = x/1e-6
    x_f = x_f/1e-6
    f, Pxx_den = signal.welch(x, fs, nperseg=2048)
    f_f, Pxx_den_f = signal.welch(x_f, fs, nperseg=2048)

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(211)
    plt.title('PSD')
    plt.plot(f, Pxx_den.mean(axis=0))
    plt.yscale('log')
    plt.xlim((0, 280))
    plt.ylim((10e-12, 10e2))
    plt.ylabel('Não Filtrado [V²/Hz] (dB)')
    plt.grid(which='both', axis='both')
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.plot(f_f, Pxx_den_f.mean(axis=0))
    plt.yscale('log')
    plt.ylabel('Filtrado [V²/Hz] (dB)')
    plt.xlabel('Frequency [Hz]')
    plt.grid(which='both', axis='both')
    plt.axvline(cut, color='green')  # cutoff frequency
    plt.show()


def my_time_fft_comparison(x, fs, canal, xmax):

    n = np.size(x, 0)
    dt = 1 / fs
    yf = fft(x, axis=0)
    xfp = np.linspace(0.0, 1.0 / (2.0 * dt), n // 2)
    yf_plot = 2.0 / n * np.abs(yf[0:n // 2])
    x = x / 1e-6
    t = np.arange(0, len(x) / fs, 1 / fs)

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(211)
    plt.plot(t, x)
    plt.ylim((-100, 100))
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [uV]')
    plt.title(canal)
    plt.xlim((0, xmax))
    plt.grid()
    ax2 = plt.subplot(212)
    plt.plot(xfp, yf_plot)
    plt.xlim((-1, 300))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Single-Sided Amplitude Spectrum |X| [V]')
    plt.grid()
    plt.show()


def my_plot_filter_response(bfilter, afilter, cut):
    plt.figure(figsize=(10, 4))
    wfreq, h = signal.freqz(bfilter, afilter)
    plt.semilogx(wfreq, 20 * np.log10(abs(h)))
    plt.title('Filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(cut, color='green')  # cutoff frequency
    plt.show()


def my_epoca(data, fs, limi1, limi2, acc_canal, aux=0):
    acc = data[:, acc_canal]
    data = data*1e-6
    t = np.arange(0, len(data) / fs, 1 / fs)

    b, a = signal.butter(2, 2 / fs, 'low')
    acc_filt = signal.filtfilt(b, a, acc)
    del b, a

    acc_diff = np.diff(acc_filt)

    flex_peaks, _ = signal.find_peaks(acc_diff, height=limi1, distance=12 * fs)

    ext_peaks, _ = signal.find_peaks(-acc_diff, height=limi2, distance=12 * fs)

    # Plot
    if aux:
        plt.figure(1, figsize=(12, 6))
        plt.subplot(211)
        plt.plot(t, acc_filt, label="filter")
        plt.plot(t, acc_diff, label="diff")
        plt.subplot(212)
        plt.plot(acc_diff)
        plt.plot(flex_peaks, acc_diff[flex_peaks], "x")
        plt.plot(ext_peaks, acc_diff[ext_peaks], "o")
        plt.show()

    atividade_pos = np.zeros([1, len(acc)])
    atividade_pos[0, flex_peaks] = 1
    atividade_pos[0, ext_peaks] = 2

    # alternative
    #flex_pos = np.in1d(t_pos, flex_peaks, assume_unique=True)
    #flex_pos = 1 * flex_pos
    #ext_pos = np.in1d(t_pos, ext_peaks, assume_unique=True)
    #ext_pos = 2 * ext_pos
    #atividade_pos = ext_pos + flex_pos

    data[:, acc_canal] = atividade_pos
    data = data.transpose()
    return data


# def rejeitaartefato(epoch_eeg, epoch_base, m_dp=3, m_percent=5, m_percent_total=10, amp=150e-6):
#
#     discarded_m_percent = np.zeros(np.shape(epoch_eeg[:, :, 0]), dtype=bool)
#     discarded_m_percent_total = np.zeros(np.shape(epoch_eeg[:, :, 0]), dtype=bool)
#     discarded_amp = np.zeros(np.shape(epoch_eeg[:, :, 0]), dtype=bool)
#     signal_discarded = []
#     signal_out = []
#     epoch_discarded = []
#
#     dp_base = np.std(epoch_eeg[epoch_base, :, :], 2)
#     dp_mean = np.mean(dp_base, 0)
#     threshold = m_dp * dp_mean
#     window_percent = np.ceil(np.size(epoch_eeg, 2) * (m_percent * 0.01))
#     window_percent_total = np.ceil(np.size(epoch_eeg, 2) * (m_percent_total * 0.01))
#
#     for ind_epoch in range(np.size(epoch_eeg, 0)):
#         for ind_ch in range(np.size(epoch_eeg, 1)):
#             sinal_teste = epoch_eeg[ind_epoch, ind_ch, :]
#             xpos = threshold[ind_ch] < sinal_teste
#             xneg = -threshold[ind_ch] > sinal_teste
#
#             dif_neg = np.diff(np.where(np.concatenate(([xneg[0]], xneg[:-1] != xneg[1:], [True])))[0])[::2]
#             dif_pos = np.diff(np.where(np.concatenate(([xpos[0]], xpos[:-1] != xpos[1:], [True])))[0])[::2]
#             total_neg = np.sum(xneg)
#             total_pos = np.sum(xpos)
#
#             discarded_m_percent[ind_epoch, ind_ch] = any(dif_neg > window_percent) or any(dif_pos > window_percent)
#             discarded_m_percent_total[ind_epoch, ind_ch] = (total_neg > window_percent_total) or (
#                         total_pos > window_percent_total)
#             discarded_amp[ind_epoch, ind_ch] = any(sinal_teste > amp) or any(sinal_teste < -amp)
#
#         if (discarded_m_percent[ind_epoch, :].any() or
#                 discarded_m_percent_total[ind_epoch, :].any() or
#                 discarded_amp[ind_epoch, :].any()):
#             signal_discarded.append(epoch_eeg[ind_epoch, :, :])
#             epoch_discarded.append(ind_epoch)
#         else:
#             signal_out.append(epoch_eeg[ind_epoch, :, :])
#
#     return np.array(signal_out), np.array(signal_discarded), np.array(epoch_discarded)
def rejeitaartefato(epoch_eeg, epoch_base, m_dp=3, m_percent=5, m_percent_total=10, amp=150e-6):

    discarded_m_percent = np.zeros(np.shape(epoch_eeg[:, :, 0]), dtype=bool)
    discarded_m_percent_total = np.zeros(np.shape(epoch_eeg[:, :, 0]), dtype=bool)
    discarded_amp = np.zeros(np.shape(epoch_eeg[:, :, 0]), dtype=bool)
    signal_discarded = []
    signal_out = []
    epoch_discarded = []

    dp_base = np.std(epoch_eeg[epoch_base, :, :], 2)
    dp_mean = np.mean(dp_base, 0)
    threshold = m_dp * dp_mean
    window_percent = np.ceil(np.size(epoch_eeg, 2) * (m_percent * 0.01))
    window_percent_total = np.ceil(np.size(epoch_eeg, 2) * (m_percent_total * 0.01))

    for ind_epoch in range(np.size(epoch_eeg, 0)):
        for ind_ch in range(np.size(epoch_eeg, 1)):
            sinal_teste = epoch_eeg[ind_epoch, ind_ch, :]
            xpos = threshold[ind_ch] < sinal_teste
            xneg = -threshold[ind_ch] > sinal_teste

            dif_neg = np.diff(np.where(np.concatenate(([xneg[0]], xneg[:-1] != xneg[1:], [True])))[0])[::2]
            dif_pos = np.diff(np.where(np.concatenate(([xpos[0]], xpos[:-1] != xpos[1:], [True])))[0])[::2]
            total_neg = np.sum(xneg)
            total_pos = np.sum(xpos)

            discarded_m_percent[ind_epoch, ind_ch] = any(dif_neg > window_percent) or any(dif_pos > window_percent)
            discarded_m_percent_total[ind_epoch, ind_ch] = (total_neg > window_percent_total) or (
                        total_pos > window_percent_total)
            discarded_amp[ind_epoch, ind_ch] = any(sinal_teste > amp) or any(sinal_teste < -amp)

        if (discarded_m_percent[ind_epoch, :].any() or
                discarded_m_percent_total[ind_epoch, :].any() or
                discarded_amp[ind_epoch, :].any()):
            signal_discarded.append(epoch_eeg[ind_epoch, :, :])
            epoch_discarded.append(ind_epoch)
        else:
            signal_out.append(epoch_eeg[ind_epoch, :, :])

    return np.array(signal_out), np.array(signal_discarded), np.array(epoch_discarded), discarded_m_percent, discarded_m_percent_total, discarded_amp

def my_epoca2(data, fs, param, acc_canal, usuario, plot):
    acc = data[:, acc_canal]
    data = data*1e-6
    order = param[0]
    h_ext = param[1]
    dt_ext = param[2]
    lim_ext = param[3]
    t_ext = param[4]
    h_flex = param[5]
    dt_flex = param[6]
    lim_flex = param[7]
    t_flex = param[8]

    b, a = signal.butter(order, 2 / fs, 'low')
    acc_filt = signal.filtfilt(b, a, acc)
    del b, a
    acc_diff = np.diff(acc_filt)
    acc_filt = acc_filt - np.mean(acc_filt)
    # Teste
    acc_diff = acc_diff - np.mean(acc_diff)

    ext_peaks, _ = signal.find_peaks(acc_diff, height=h_ext, distance=dt_ext * fs)
    new_ext = np.delete(ext_peaks, [np.where(acc_diff[ext_peaks] > lim_ext)])
    new_ext2 = new_ext[(acc_filt[new_ext] > 0)]
    aux = np.delete(new_ext2, [np.where(new_ext2 + int(t_ext * fs+1) > len(acc_filt))])
    ext_trigger = aux[(acc_filt[aux + int(t_ext * fs)] < 0)]

    flex_peaks, _ = signal.find_peaks(-acc_diff, height=-h_flex, distance=dt_flex * fs)
    new_flex = np.delete(flex_peaks, [np.where(acc_diff[flex_peaks] < -lim_flex)])
    new_flex2 = new_flex[(acc_filt[new_flex] < 0)]
    aux = np.delete(new_flex2, [np.where(new_flex2 + int(t_flex * fs +1) > len(acc_filt))])
    new_flex3 = aux[(acc_filt[aux + int(t_flex * fs)] > 0)]
    mask = np.where(np.diff(new_flex3) < 1000)
    flex_trigger = np.delete(new_flex3, mask)

    '''
    #ext_trigger = detect_peaks(acc_diff, mph=limi1, mpd=dt1 * fs, edge='rising')
    #flex_trigger = detect_peaks(acc_diff, mph=limi2, mpd=dt2 * fs, edge='falling', valley=True, threshold=0)
    ext_peaks, _ = signal.find_peaks(acc_diff, height=ext1, distance=2 * fs)
    ext_aux = np.delete(ext_peaks, [np.where(acc_diff[ext_peaks] > ext2)])
    ext_trigger = ext_aux[(acc_filt[ext_aux] > 0)]

    flex_peaks, _ = signal.find_peaks(-acc_diff, height=flex1, distance=2 * fs)
    flex_aux = np.delete(flex_peaks, [np.where(acc_diff[flex_peaks] < -flex2)])
    flex_trigger = flex_aux[(acc_filt[flex_aux] < 0)]
    '''
    # Plot
    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(acc_filt/5e2, label="filter")
        ax.plot(acc_diff, label="diff")
        ax.plot(flex_trigger, acc_diff[flex_trigger], "x", label="flex")
        ax.plot(ext_trigger, acc_diff[ext_trigger], "o", label="ext")
        ax.grid()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        plt.legend(loc='upper right')
        plt.show()


    atividade_pos = np.zeros([1, len(acc)])
    atividade_pos[0, flex_trigger] = 1
    atividade_pos[0, ext_trigger] = 2

    data[:, acc_canal] = atividade_pos

    if usuario == 1:
        data[:, acc_canal + 1] = acc_filt # Lana
    elif usuario == 2:
        data[:, acc_canal - 1] = acc_filt  # Alexsandro
    data = data.transpose()
    return data

def my_epoca3(data, fs, r, param, acc_canal, usuario, plot, teste):
    #Decimate
    acc = data[:, acc_canal]
    data = data*1e-6
    order = param[0]
    h_ext = param[1]
    dt_ext = param[2]
    lim_ext = param[3]
    t_ext = param[4]
    h_flex = param[5]
    dt_flex = param[6]
    lim_flex = param[7]
    t_flex = param[8]

    acc0 = signal.decimate(acc, r)
    fs_decimate = fs / r
    b, a = signal.butter(order, 2 / fs_decimate, 'low')
    acc_filt0 = signal.filtfilt(b, a, acc0)
    del b, a
    acc_diff0 = np.diff(acc_filt0)
    acc_filt0 = acc_filt0 - np.mean(acc_filt0)
    # Teste
    acc_diff0 = acc_diff0 - np.mean(acc_diff0)

    ext_peaks, _ = signal.find_peaks(acc_diff0, height=h_ext, distance=dt_ext * fs_decimate)
    new_ext = np.delete(ext_peaks, [np.where(acc_diff0[ext_peaks] > lim_ext)])
    new_ext2 = new_ext[(acc_filt0[new_ext] > 0)]
    aux = np.delete(new_ext2, [np.where(new_ext2 + int(t_ext * fs_decimate+1) > len(acc_filt0))])
    ext_trigger0 = aux[(acc_filt0[aux + int(t_ext * fs_decimate)] < 0)]

    flex_peaks, _ = signal.find_peaks(-acc_diff0, height=-h_flex, distance=dt_flex * fs_decimate)
    new_flex = np.delete(flex_peaks, [np.where(acc_diff0[flex_peaks] < -lim_flex)])
    new_flex2 = new_flex[(acc_filt0[new_flex] < 0)]
    aux = np.delete(new_flex2, [np.where(new_flex2 + int(t_flex * fs_decimate +1) > len(acc_filt0))])
    new_flex3 = aux[(acc_filt0[aux + int(t_flex * fs_decimate)] > 0)]
    mask = np.where(np.diff(new_flex3) < 1000)
    flex_trigger0 = np.delete(new_flex3, mask)

    if teste:
        b, a = signal.butter(order, 2 / fs, 'low')
        acc_filt = signal.filtfilt(b, a, acc)
        del b, a
        acc_diff = np.diff(acc_filt)
        acc_filt = acc_filt - np.mean(acc_filt)
        # Teste
        acc_diff = acc_diff - np.mean(acc_diff)

        ext_peaks, _ = signal.find_peaks(acc_diff, height=h_ext, distance=dt_ext * fs)
        new_ext = np.delete(ext_peaks, [np.where(acc_diff[ext_peaks] > lim_ext)])
        new_ext2 = new_ext[(acc_filt[new_ext] > 0)]
        aux = np.delete(new_ext2, [np.where(new_ext2 + int(t_ext * fs+1) > len(acc_filt))])
        ext_trigger = aux[(acc_filt[aux + int(t_ext * fs)] < 0)]

        flex_peaks, _ = signal.find_peaks(-acc_diff, height=-h_flex, distance=dt_flex * fs)
        new_flex = np.delete(flex_peaks, [np.where(acc_diff[flex_peaks] < -lim_flex)])
        new_flex2 = new_flex[(acc_filt[new_flex] < 0)]
        aux = np.delete(new_flex2, [np.where(new_flex2 + int(t_flex * fs +1) > len(acc_filt))])
        new_flex3 = aux[(acc_filt[aux + int(t_flex * fs)] > 0)]
        mask = np.where(np.diff(new_flex3) < 1000)
        flex_trigger = np.delete(new_flex3, mask)

    '''
    #ext_trigger = detect_peaks(acc_diff, mph=limi1, mpd=dt1 * fs, edge='rising')
    #flex_trigger = detect_peaks(acc_diff, mph=limi2, mpd=dt2 * fs, edge='falling', valley=True, threshold=0)
    ext_peaks, _ = signal.find_peaks(acc_diff, height=ext1, distance=2 * fs)
    ext_aux = np.delete(ext_peaks, [np.where(acc_diff[ext_peaks] > ext2)])
    ext_trigger = ext_aux[(acc_filt[ext_aux] > 0)]

    flex_peaks, _ = signal.find_peaks(-acc_diff, height=flex1, distance=2 * fs)
    flex_aux = np.delete(flex_peaks, [np.where(acc_diff[flex_peaks] < -flex2)])
    flex_trigger = flex_aux[(acc_filt[flex_aux] < 0)]
    '''
    # Plot
    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(acc_filt/5e2, label="filter")
        ax.plot(acc_diff, label="diff")
        ax.plot(flex_trigger0, acc_diff[flex_trigger0], "x", label="flex")
        ax.plot(ext_trigger0, acc_diff[ext_trigger0], "o", label="ext")
        ax.grid()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        plt.legend(loc='upper right')
        plt.show()


    atividade_pos = np.zeros([1, len(acc0)])
    atividade_pos[0, flex_trigger0] = 1
    atividade_pos[0, ext_trigger0] = 2

    data[:, acc_canal] = atividade_pos

    if usuario == 1:
        data[:, acc_canal + 1] = acc_filt0 # Lana
    elif usuario == 2:
        data[:, acc_canal - 1] = acc_filt0  # Alexsandro
    data = data.transpose()

    if teste:
        print('ext:', len(ext_trigger), len(ext_trigger0))
        print('flex:', len(flex_trigger), len(flex_trigger0))

    return data
