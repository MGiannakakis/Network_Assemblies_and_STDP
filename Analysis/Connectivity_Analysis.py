import numpy as np
import itertools

from brian2 import *
from Neuron_response import *

def get_mean_and_dist_weights(We, Wi, NSigs = 8):
    
    NE = len(We)
    NI = len(Wi)
    
    W_Inh = []
    W_Ex = []
    dist_wi = 0
    dist_we = 0

    for g in range(NSigs):
        we = We[int(NE / NSigs) * g:int(NE / NSigs) * (g + 1)]
        dist_we += np.var(we) / (NSigs * np.var(We))
        wge = np.mean(we)
        W_Ex.append(wge)

        wi = Wi[int(NI / NSigs) * g:int(NI / NSigs) * (g + 1)]
        dist_wi += np.var(wi) / (NSigs * np.var(Wi))
        wgi = np.mean(wi)
        W_Inh.append(wgi)
        
    return [W_Ex, W_Inh, dist_we, dist_wi]

    

def get_conn_metrics(We, Wi, NSigs = 8):
    
    [W_Ex, W_Inh, dist_we, dist_wi] = get_mean_and_dist_weights(We, Wi, NSigs = NSigs)
    
    CT_W = round(np.corrcoef(W_Ex, W_Inh[::-1])[0, 1], 4)
    D = (round(1 - dist_we, 4) + round(1 - dist_wi, 4))/2

    return CT_W, D
    
    
def get_W_FR_Cor(We, Wi, NSigs = 8):
    
    [W_Ex, W_Inh, _, _] = get_mean_and_dist_weights(We, Wi, NSigs = NSigs)
    Rates_tuned = get_tunings(np.array(We), np.array(Wi), NSigs = NSigs)
    
    return np.corrcoef(W_Ex, np.mean(Rates_tuned, axis = 0))[0, 1]