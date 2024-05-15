import numpy as np
from brian2 import *

def get_cur_volt(mon, v_e = 0*mV, v_i = -80*mV):
    V = mon.v 
    Ampa = mon.g_ampa
    Gaba = mon.g_gaba

    E_ex = -Ampa*(V - v_e)
    E_in = -Gaba*(V - v_i)

    ex = E_ex[0]
    inh = E_in[0]
    cur = ex + inh
       
    return ex, inh, cur



def get_balance(mon, v_e = 0*mV, v_i = -80*mV):
    
    ex, inh, cur = get_cur_volt(mon, v_e = v_e, v_i = v_i)
    
    Balance = 2*cur/(np.abs(ex) + np.abs(inh))
    
    return Balance


