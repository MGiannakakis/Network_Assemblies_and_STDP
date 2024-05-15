import numpy as np
import itertools

from brian2 import *

def get_cor(Sp, NSigs):
    sp = np.array(Sp)
    S = np.zeros((2*NSigs, 2*NSigs))
    for i in range(2*NSigs):
        for j in range(2*NSigs):
            if (np.std(sp[i]) > 0 and np.std(sp[j]) > 0):
                S[i][j] = np.corrcoef(sp[i], sp[j])[0, 1]
            else:
                if i == j or np.abs(i - j) == NSigs:
                    S[i][j] = 1
                else:
                    S[i][j] = 0
    return S


def get_filtered_spikes(spikes, tau):
    Y = [0]
    for s in spikes:
        Y.append(Y[-1] + (-Y[-1] + s)/tau)
    return Y

def get_corrs(sm, NE = 800, NI = 200, NSigs = 8):

    i, t = sm.it
    (he,be) = np.histogram(i[i<NE], NSigs, range = (0, NE))
    (hi,bi) = np.histogram(i[i>=NE], NSigs, range = (NE, NE + NI))
        
    SpE = []
    SpI = []
    i, t = sm.it
    t = t / ms
    t0, tmax = 0, np.max(t)
    bins_size = 1

    # trains
    for g in range(NSigs):
        Emask = (i >= int(NE / NSigs) * g) * (i < int(NE / NSigs) * (g + 1))
        Imask = (i >= NE + int(NI / NSigs) * (NSigs - g - 1)) * (i < NE + int(NI / NSigs) * (NSigs - g))
        scE, bins = np.histogram(np.array(t)[Emask], np.arange(t0, tmax, bins_size))
        scI, bins = np.histogram(np.array(t)[Imask], np.arange(t0, tmax, bins_size))

        SpE.append(scE)
        SpI.append(scI)

    Sp = SpE + SpI
    S = get_cor(Sp, NSigs)

    EI_IN = np.mean([S[i][i + NSigs] for i in range(NSigs)])
    EI_OUT = np.mean([S[i][j + NSigs] for (i, j) in list(itertools.combinations(range(NSigs), 2)) if i != j])
    EE = np.mean([S[i][j] for (i, j) in list(itertools.combinations(range(NSigs), 2))])
    II = np.mean([S[i + NSigs][j + NSigs] for (i, j) in list(itertools.combinations(range(NSigs), 2))])

    return [EI_IN, EI_OUT, EE, II]


def get_stats(sm):
    spikes = sm.spike_trains()
    si,st = sm.it
    t = st / ms
    t0, tmax = 50, np.max(t)
    bins_size = 1

    ISI_mean = []
    CV =[]
    FF = []

    for i in spikes.keys():
        ISI_mean.append(np.mean(np.diff(spikes[i]/second)))
        CV.append(np.std(np.diff(spikes[i]/second))/np.mean(np.diff(spikes[i]/second)))
        scE, _ = np.histogram(spikes[i]/ms, np.arange(t0, tmax, bins_size))
        FF.append(np.var(scE)/np.mean(scE))
        
    return [ISI_mean, CV, FF]