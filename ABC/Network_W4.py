from brian2 import *

########################
# Connectivity Functions
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


def get_out_group_connectivity(pre, post, prob):
    Out = []
    In = []
    for i in range(len(pre)):
        for j in range(len(post)):
           # if pre[i] != post[j]:
                if rand() < prob:
                    Out.append(i)
                    In.append(j)


#####################################
# Low pass filter
#####################################
def get_filtered_spikes(spikes, tau):
    Y = [0]
    for s in spikes:
        Y.append(Y[-1] + (-Y[-1] + s) / tau)
    return Y

##########################
# Clustering Function
#########################
def get_con_params(NSigs, r, V):
    d = (NSigs - 1) * (r + 1)

    v_in = (NSigs * r - r - 1) * NSigs * V / d
    v_out = NSigs * V / d

    return v_in, v_out


def get_cor(Sp, NSigs):
    sp = np.array(Sp)
    S = np.zeros((2*NSigs, 2*NSigs))
    for i in range(2*NSigs):
        for j in range(2*NSigs):
            if (np.std(sp[i]) > 0 and np.std(sp[j]) > 0):
                S[i][j] = np.corrcoef(sp[i], sp[j])[0, 1]
            else:
                if i == j or np.abs(i - j) == NSigs:
                    S[i][j] = 0.999
                else:
                    S[i][j] = 0.0001
    return S


#####################################
# Run Network
#####################################


def Network_run(W = 0.0,
                p = 0.0,#connection proability
                noise_ratio = 0.01, 
                R = [1., 1., 1., 1.],#assembly strength
                N=1000,#num of neurons
                tmax=25.):#sim time

    ########################
    # Simulation Parameters
    ########################

    NSigs = 8  # Number of input signals.
    simtime = tmax * second #


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

    NCells = N  # Number of input spike trains.
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


    #######################################
    # Recurrent connections Between Groups
    #######################################
    
    i_s = gBarEx/(gBarIn*(1-ExFrac))


    if p > 0:
        R_ee = Synapses(Pe, Pe, model='w: 1', on_pre='g_ampa += w*gBarEx*nS')
        I, J = get_out_group_connectivity(InputGroup[:NE], InputGroup[:NE], p)
        R_ee.connect(i=I, j=J)
        R_ee.w = w_ee_out + 1e-1*w_ee_out*randn(len(R_ee.w))
    
        R_ei = Synapses(Pe, Pi, model='w: 1', on_pre='g_ampa += w*gBarEx*nS')
        I, J = get_out_group_connectivity(InputGroup[:NE], InputGroup[NE:], p)
        R_ei.connect(i=I, j=J)
        R_ei.w = w_ei_out + 1e-1*w_ei_out*randn(len(R_ei.w))
    
        R_ie = Synapses(Pi, Pe, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        I, J = get_out_group_connectivity(InputGroup[NE:], InputGroup[:NE], p)
        R_ie.connect(i=I, j=J)
        R_ie.w = w_ie_out + 1e-1*w_ie_out*randn(len(R_ie.w))
    
        R_ii = Synapses(Pi, Pi, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        I, J = get_out_group_connectivity(InputGroup[NE:], InputGroup[NE:], p)
        R_ii.connect(i=I, j=J)
        R_ii.w = w_ii_out + 1e-1*w_ii_out*randn(len(R_ii.w))

    #######################################
    # Recurrent connections Within Groups
    #######################################

    if p > 0:
        R_ee_group = Synapses(Pe, Pe, model='w: 1', on_pre='g_ampa += w*gBarEx*nS')
        I, J = get_in_group_connectivity(InputGroup[:NE], InputGroup[:NE], p)
        R_ee_group.connect(i=I, j=J)
        R_ee_group.w = w_ee_in + 0.1*w_ee_in*randn(len(R_ee_group.w))

        R_ei_group = Synapses(Pe, Pi, model='w: 1', on_pre='g_ampa += w*gBarEx*nS')
        I, J = get_in_group_connectivity(InputGroup[:NE], InputGroup[NE:], p)
        R_ei_group.connect(i=I, j=J)
        R_ei_group.w = w_ei_in + 0.1*w_ei_in*randn(len(R_ei_group.w))

        R_ie_group = Synapses(Pi, Pe, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        I, J = get_in_group_connectivity(InputGroup[NE:], InputGroup[:NE], p)
        R_ie_group.connect(i=I, j=J)
        R_ie_group.w = w_ie_in + 0.1*w_ie_in*randn(len(R_ie_group.w))

        R_ii_group = Synapses(Pi, Pi, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        I, J = get_in_group_connectivity(InputGroup[NE:], InputGroup[NE:], p)
        R_ii_group.connect(i=I, j=J)
        R_ii_group.w = w_ii_in + 0.1*w_ii_in*randn(len(R_ii_group.w))

    # ###########################################
    # Run Simulation without Plasticity
    # ###########################################

    # Spike Monitors
    sm = SpikeMonitor(P)
    run(simtime)
    
    
   # FR Var Between groups

    i, t = sm.it
    (he,be) = np.histogram(i[i<NE], NSigs, range = (0, NE))
    (hi,bi) = np.histogram(i[i>=NE], NSigs, range = (NE, NE + NI))

    g_var = 0.5*(np.std(he)/np.mean(he) + np.std(hi)/np.mean(hi))

    ########################################
    # Correlations
    ########################################
    import itertools
    SpE = []
    SpI = []
    i, t = sm.it
    t = t / ms
    t0, tmax = 1, np.max(t)
    bins_size = 1

    # trains
    n_active_u = []
    for g in range(NSigs):
        Emask = (i >= int(NE / NSigs) * g) * (i < int(NE / NSigs) * (g + 1))
        Imask = (i >= NE + int(NI / NSigs) * (NSigs - g - 1)) * (i < NE + int(NI / NSigs) * (NSigs - g))
        scE, bins = np.histogram(np.array(t)[Emask], np.arange(t0, tmax, bins_size))
        scI, bins = np.histogram(np.array(t)[Imask], np.arange(t0, tmax, bins_size))
        
        # n_active_u.append(np.sum(scE)/ExGroupsize)

        SpE.append(scE)
        SpI.append(scI)

    Sp = SpE + SpI
    S = get_cor(Sp, NSigs)

    EI_IN = np.mean([S[i][i + NSigs] for i in range(NSigs)])
    EI_OUT = np.mean([S[i][j + NSigs] for (i, j) in list(itertools.combinations(range(NSigs), 2)) if i != j])
    EE = np.mean([S[i][j] for (i, j) in list(itertools.combinations(range(NSigs), 2))])
    II = np.mean([S[i + NSigs][j + NSigs] for (i, j) in list(itertools.combinations(range(NSigs), 2))])

    # Return Correlation

    return [EI_IN, EI_OUT, EE, II, g_var]
