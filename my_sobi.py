'''
'''

import numpy as np
from scipy import linalg as LA
from scipy import stats
import matplotlib.pyplot as plt
import math
from matplotlib.widgets import Slider
import mne


def my_sobi(X):
    N = np.size(X,0)
    m = np.size(X, 1)

    X_mean = X.mean(axis=0)
    X -= X_mean

    # Pre-whiten the data based directly on SVD
    u, d, v = LA.svd(X, full_matrices=False, check_finite=False)
    d = np.diag(d) #d -> matriz diagonal
    Q = np.dot(LA.pinv(d), v)
    X1 = np.dot(Q, X.T)

    p = min(100, math.ceil(N / 3))

    pm = p * m
    Rxp = np.zeros((m, m))
    M = np.zeros((m, pm), dtype=complex)

    # Estimate the correlation matrices
    k = 0

    for u in range(0, pm - 1, m):
        k += 1
        Rxp = np.dot(X1[:, k:N], X1[:, 0:N - k].T) / (N - k)
        M[:, u:u + m] = LA.norm(Rxp, 'fro') * Rxp

    # Perform joint diagonalization
    epsil = 1 / math.sqrt(N) / 100
    encore = 1
    V = np.eye(m, dtype=complex)
    g = np.zeros((m, p))

    while encore:
        encore = 0
        for p_ind in range(0, m - 1):
            for q_ind in range(p_ind + 1, m):
                g = [M[p_ind, p_ind:pm + 1:m] - M[q_ind, q_ind:pm + 1:m],
                     M[p_ind, q_ind:pm + 1:m] + M[q_ind, p_ind:pm + 1:m],
                     1j * (M[q_ind, p_ind:pm + 1:m] - M[p_ind, q_ind:pm + 1:m])]
                g = np.array(g)
                z = np.real(np.dot(g, g.T))
                w, vr = LA.eig(z, left=False, right=True)
                K = np.argsort(abs(w))
                temp_ang = vr[:, K[2]]
                angles = np.sign(temp_ang[0]) * temp_ang
                c = np.sqrt(0.5 + angles[0] / 2)
                sr = 0.5 * (angles[1] - 1j * angles[2]) / c
                sc = np.conj(sr)
                oui = np.abs(sr) > epsil
                encore = encore | oui

                if oui:
                    temp_M = np.copy(M)
                    colp = temp_M[:, p_ind:pm + 1:m]
                    colq = temp_M[:, q_ind:pm + 1:m]
                    M[:, p_ind:pm + 1:m] = c * colp + sr * colq
                    M[:, q_ind:pm + 1:m] = c * colq - sc * colp

                    temp_M2 = np.copy(M)
                    rowp = temp_M2[p_ind, :]
                    rowq = temp_M2[q_ind, :]
                    M[p_ind, :] = c * rowp + sc * rowq
                    M[q_ind, :] = c * rowq - sr * rowp

                    temp_V = np.copy(V)
                    V[:, p_ind] = c * temp_V[:, p_ind] + sr * temp_V[:, q_ind]
                    V[:, q_ind] = c * temp_V[:, q_ind] - sc * temp_V[:, p_ind]

    # Estimate the mixing matrix
    H = np.dot(LA.pinv(Q), V)

    # Estimated source activities
    Source = np.dot(V.T, X1)

    return H, Source


def comp_plot(x, x1, fs, ylim ,ch_name, title):

    t = np.arange(0, np.size(x, 1) / fs, 1 / fs)
    a = np.size(x, 0)
    b = 1

    xlim = [0, 20]
    fig, axes = plt.subplots(a, b, sharex=True, sharey=True, figsize=(10, 8))
    fig.suptitle(title)
    for ind, ax in enumerate(axes.flatten()):
        ax.plot(t, x[ind, :].real, 'r-', linewidth=1)
        ax.plot(t, x1[ind, :], 'k', linewidth=1)
        ax.set(ylabel=ch_name[ind])
        ax.set_xlim(xlim)
        ax.set_ylim([-ylim, ylim])
        ax.grid()
    plt.show()


def ica_plot(x, eog, fs, IC_name, corrcoef, title, escala):

    t = np.arange(0, np.size(x, 1) / fs, 1 / fs)
    a = np.size(x, 0)
    xlim = [0, 20]
    ylim = [-200, 200]
    fig, axes = plt.subplots(nrows=a + 1, sharex=True, sharey=True, figsize=(10, 8))
    fig.suptitle(title)

    for ind, ax in enumerate(axes.flatten()):
        if ind == a:
            ax.plot(t, eog*escala*50, 'r', linewidth=1)
            ax.set(ylabel='eog')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.yaxis.set_ticklabels([])
            ax.grid()
        else:
            ax.plot(t, x[ind, :]*escala, 'k', linewidth=1)
            aux = str(IC_name[ind]) + '\n (' + ('%.2f' % corrcoef[ind]) +')'
            ax.set(ylabel=aux)
            ax.set_ylim(ylim)
            ax.yaxis.set_ticklabels([])
            ax.grid()

    plt.show()


def my_ica(raw):
    eeg_ica = raw.get_data(picks='eeg')
    eog = raw.get_data(picks='eog')

    dp = np.std(eeg_ica, 1)
    eeg_ica = eeg_ica.T
    eeg_ica = np.divide(eeg_ica, dp)

    H, S = my_sobi(eeg_ica)
    H = H.real
    S = S.real

    corrp = np.zeros((2, np.size(S, 0)))
    for ind in range(np.size(S, 0)):
        teste = S[ind, :]
        teste = teste.astype(float)
        teste = teste.flatten()
        eog = eog.T
        eog = eog.flatten()
        corrp[:, ind] = stats.pearsonr(teste, eog)

    print(corrp[0, :])
    componente = np.nanargmax(np.absolute(corrp[0, :]))
    print(componente)
    H0 = np.copy(H)
    H0[:, componente] = np.zeros((np.size(H, 0)))
    eeg_recon = H0 @ S
    eeg_recon = eeg_recon * dp[:, None]

    return eeg_recon, S, corrp


def ICA_comp_plot(raw, S, corrp, maxch=9):

    eog = raw.get_data(picks='eog').squeeze()
    fs = raw.info['sfreq']

    indplot = -(-np.size(S, 0) // maxch)
    IC_name = np.arange(0, np.size(S, 0))

    for auxplot in range(indplot):

        if auxplot < indplot:
            S_plot = S[auxplot * maxch:(auxplot + 1) * maxch, :]
            corrcoef = corrp[0, auxplot * maxch:(auxplot + 1) * maxch]
            IC = IC_name[auxplot * maxch:(auxplot + 1) * maxch]

        else:
            S_plot = S[auxplot * maxch:, :]
            corrcoef = corrp[0, auxplot * maxch:]
            IC = IC_name[auxplot * maxch:, :]

        ica_plot(S_plot, eog, fs, IC, corrcoef, 'ICA ' + str(auxplot), 2e4)


def recon_plot(raw, eeg_recon, maxch=9):
    eeg = raw.get_data(picks='eeg')
    fs = raw.info['sfreq']
    ch_names = raw.info['ch_names']
    indplot = -(-np.size(eeg, 0) // maxch)
    for auxplot in range(indplot):

        if auxplot < indplot:
            eeg_recon_plot = eeg_recon[auxplot * maxch:(auxplot + 1) * maxch, :]
            eeg_plot = eeg[auxplot * maxch:(auxplot + 1) * maxch, :]
            ch = ch_names[auxplot * maxch:(auxplot + 1) * maxch]

        else:
            eeg_recon_plot = eeg_recon[auxplot * maxch:, :]
            eeg_plot = eeg[0, auxplot * maxch:]
            ch = ch_names[auxplot * maxch:, :]

        comp_plot(eeg_recon_plot, eeg_plot, fs, 1e-4, ch, 'Comparacao ' + str(auxplot))




#    recon_raw = raw.copy()
#    recon_raw['eeg'] = eeg_recon


'''
np.random.seed(0)
N = 6

time = np.linspace(0, 4, N)
f1 = 10
f2 = 27
s1 = np.sin(2*math.pi*f1 * time) # Signal 1 : sinusoidal signal
s2 = np.sin(2*math.pi*f2 * time) # Signal 1 : sinusoidal signal
s3 = np.ones(N)
S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)


A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix

X = np.dot(S, A.T)  # amostras x variaveis
H, source = my_sobi(X)
'''


