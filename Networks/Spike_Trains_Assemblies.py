from brian2 import *
from .Connectivity_Utils import *


########################
# Connectivity Function
########################
def get_in_group_connectivity(pre, post, prob):
    Out = []
    In = []
    for i in range(len(pre)):
        for j in range(len(post)):
            if pre[i] == post[j]:
                if rand() < prob:
                    Out.append(i)
                    In.append(j)
    return Out, In


##########################
# Clustering Function
#########################
def get_con_params(NSigs, r, V):
    d = (NSigs - 1) * (r + 1)

    v_in = (NSigs * r - r - 1) * NSigs * V / d
    v_out = NSigs * V / d

    return v_in, v_out

#####################################
# Run Network
#####################################


def get_spikes(NSigs = 8,
               W = 0.0,
                p = 0.0,
                noise_ratio = 0.01,
                R = [1., 1., 1., 1.]):

    simtime = 30. * second  # Run time



    #Clustering Parameters
    r_ee = R[0] * 1 / (NSigs - 1)
    r_ei = R[1] * 1 / (NSigs - 1)
    r_ie = R[2] * 1 / (NSigs - 1)
    r_ii = R[3] * 1 / (NSigs - 1)


    # Connection Probabilities In/Out group
    w_ee_in, w_ee_out = get_con_params(NSigs, r_ee, W)
    w_ei_in, w_ei_out = get_con_params(NSigs, r_ei, W)

    w_ie_in, w_ie_out = get_con_params(NSigs, r_ie, W)
    w_ii_in, w_ii_out = get_con_params(NSigs, r_ii, W)

    tau_ampa = 5.0 * ms  # Glutamatergic synaptic time constant
    tau_gaba = 10.0 * ms  # GABAergic synaptic time constant
    gl = 10.0 * nsiemens  # Leak conductance
    el = -60 * mV  # Resting potential
    er = -80 * mV  # Inhibitory reversal potential
    vt = -50 * mV  # Theshold
    memc = 200.0 * pfarad  # Membrane capacitance
    gBarEx = 0.14
    gBarIn = 0.35

    #########
    # Input
    #########

    NCells = 1000  # Number of input spike trains.
    ExFrac = 0.8  # Fraction of Excitatory spike trains.

    # Number of spike trains in each excitatory input group
    ExGroupsize = round((NCells * ExFrac) / NSigs)
    # Number of spike trains in each inhibitory input group
    InGroupsize = round((NCells * (1 - ExFrac)) / NSigs)

    ###################################
    # Input Groups
    ##################################

    temptype = -1
    InputGroup = np.zeros(NCells)

    for ind in range(NCells):
        if (ind <= NCells * ExFrac):
            if ((ind - 1) % ExGroupsize == 0):
                temptype += 1
            InputGroup[ind] = temptype
        else:
            if (ind % InGroupsize == 0):
                temptype -= 1
            InputGroup[ind] = temptype

    InputGroup[0] = InputGroup[1]

    # ###########################################
    # Create Incoming Spike Trains
    # ###########################################

    signal_rate = (1-noise_ratio)*50
    noise_rate = noise_ratio*50

    Signal = PoissonGroup(NSigs, signal_rate * Hz)
    Noise = PoissonGroup(NCells, noise_rate * Hz)

    eqR = '''
    dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er)))/memc : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gaba/dt = -g_gaba/tau_gaba : siemens
    '''

    P = NeuronGroup(NCells, model=eqR, threshold='v > vt', reset='v = el', refractory=5 * ms, method='euler')

    # Neural Populations
    NE = int(NCells * ExFrac)
    NI = NCells - NE
    Pe = P[:NE]
    Pi = P[NE:]

    ################################
    # Signal feedforward connections
    ################################
    w_input = 35
    S = Synapses(Signal, P, on_pre='g_ampa += w_input*gBarEx*nS')
    S.connect(i=InputGroup.astype(int), j=range(NCells))

    Noise_Synapses = Synapses(Noise, P, on_pre='g_ampa += w_input*gBarEx*nS')
    Noise_Synapses.connect(condition='i==j')

    ########################
    # Recurrent connections 
    ########################
    
    # Inhibitory scaling
    i_s = gBarEx/(gBarIn*(1-ExFrac))
    
    
    if p > 0:
        R_ee = Synapses(Pe, Pe, model='w: 1', on_pre='g_ampa += w*gBarEx*nS')
        I, J, W = get_recurrent_connectivity(InputGroup[:NE], InputGroup[:NE],
                                             p, p, w_ee_in, w_ee_out)
        R_ee.connect(i=I, j=J)
        R_ee.w = W
    
        R_ei = Synapses(Pe, Pi, model='w: 1', on_pre='g_ampa += w*gBarEx*nS')
        I, J, W = get_recurrent_connectivity(InputGroup[:NE], InputGroup[NE:],
                                             p, p, w_ei_in, w_ei_out)
        R_ei.connect(i=I, j=J)
        R_ei.w = W
    
        R_ie = Synapses(Pi, Pe, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        I, J, W = get_recurrent_connectivity(InputGroup[NE:], InputGroup[:NE],
                                             p, p, w_ie_in, w_ie_out)
        R_ie.connect(i=I, j=J)
        R_ie.w = W
    
        R_ii = Synapses(Pi, Pi, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        I, J, W = get_recurrent_connectivity(InputGroup[NE:], InputGroup[NE:],
                                             p, p, w_ii_in, w_ii_out)
        R_ii.connect(i=I, j=J)
        R_ii.w = W

    # ###########################################
    # Run Simulation without Plasticity
    # ###########################################

    # Spike Monitors
    # Spike Monitors
    S_e = SpikeMonitor(Pe)
    S_i = SpikeMonitor(Pi)
    sm = SpikeMonitor(P)

    run(simtime, report='text')
    i, t = sm.it
    (he,be) = np.histogram(i[i<NE], NSigs, range = (0, NE))
    (hi,bi) = np.histogram(i[i>=NE], NSigs, range = (NE, NE + NI))

    g_var = 0.5*(np.std(he)/np.mean(he) + np.std(hi)/np.mean(hi))
    print(g_var)
    return [S_e.it, S_i.it]

