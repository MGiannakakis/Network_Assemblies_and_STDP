from brian2 import *
from Connectivity_Utils import *


def Network_get_Stats(W = 0.0,
                p = 0.0,
                noise_ratio = 0.01,
                R = [1., 1., 1., 1.]):

    # #############
    # Neuron model
    # #############
    NSigs = 8
    Assemblies = True

    #Clustering Parameters
    r_ee = R[0] * 1 / (NSigs - 1)
    r_ei = R[1] * 1 / (NSigs - 1)
    r_ie = R[2] * 1 / (NSigs - 1)
    r_ii = R[3] * 1 / (NSigs - 1)

    tau_ampa = 5.0 * ms  # Glutamatergic synaptic time constant
    tau_gaba = 10.0 * ms  # GABAergic synaptic time constant

    simtime = 20 * second  # Simulation time

    gl = 10.0 * nsiemens  # Leak conductance
    el = -60 * mV  # Resting potential
    er = -80 * mV  # Inhibitory reversal potential
    vt = -50 * mV  # Theshold
    memc = 200.0 * pfarad  # Membrane capacitance

    gBarEx = 0.14
    gBarIn = 0.35


    if Assemblies:
        p_ee_in = p
        p_ee_out = p
        p_ei_in = p
        p_ei_out = p
        p_ie_in = p
        p_ie_out = p
        p_ii_in = p
        p_ii_out = p

        w_ee_in, w_ee_out = get_con_params(NSigs, r_ee, W)
        w_ei_in, w_ei_out = get_con_params(NSigs, r_ei, W)

        w_ie_in, w_ie_out = get_con_params(NSigs, r_ie, W)
        w_ii_in, w_ii_out = get_con_params(NSigs, r_ii, W)

    else:
        w_ee_in = W
        w_ee_out = W
        w_ie_in = W
        w_ie_out = W
        w_ei_in = W
        w_ei_out = W
        w_ii_in = W
        w_ii_out = W

        p_ee_in, p_ee_out = get_con_params(NSigs, r_ee, p)
        p_ei_in, p_ei_out = get_con_params(NSigs, r_ei, p)

        p_ie_in, p_ie_out = get_con_params(NSigs, r_ie, p)
        p_ii_in, p_ii_out = get_con_params(NSigs, r_ii, p)

    #########
    # Input
    #########

    NCells = 1000  # Number of input spike trains.
    ExFrac = 0.8  # Fraction of Excitatory spike trains.
   # NSigs = 8  # Number of input signals.

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
    sumw :1
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


    if p_ee_out > 0:
        R_ee = Synapses(Pe, Pe, model='w: 1', on_pre='g_ampa += w*gBarEx*nS')
        I, J = get_out_group_connectivity(InputGroup[:NE], InputGroup[:NE], p_ee_out)
        R_ee.connect(i=I, j=J)
        R_ee.w = w_ee_out + 0.1*w_ee_out*randn(len(R_ee.w))

    if p_ei_out > 0:
        R_ei = Synapses(Pe, Pi, model='w: 1', on_pre='g_ampa += w*gBarEx*nS')
        I, J = get_out_group_connectivity(InputGroup[:NE], InputGroup[NE:], p_ei_out)
        R_ei.connect(i=I, j=J)
        R_ei.w = w_ei_out + 0.1*w_ei_out*randn(len(R_ei.w))

    if p_ie_out > 0:
        R_ie = Synapses(Pi, Pe, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        I, J = get_out_group_connectivity(InputGroup[NE:], InputGroup[:NE], p_ie_out)
        R_ie.connect(i=I, j=J)
        R_ie.w = w_ie_out + 0.1*w_ie_out*randn(len(R_ie.w))

    if p_ii_out > 0:
        R_ii = Synapses(Pi, Pi, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        I, J = get_out_group_connectivity(InputGroup[NE:], InputGroup[NE:], p_ii_out)
        R_ii.connect(i=I, j=J)
        R_ii.w = w_ii_out + 0.1*w_ii_out*randn(len(R_ii.w))
      
    #######################################
    # Recurrent connections Within Groups
    #######################################

    if p_ee_in > 0:
        R_ee_group = Synapses(Pe, Pe, model='w: 1', on_pre='g_ampa += w*gBarEx*nS')
        I, J = get_in_group_connectivity(InputGroup[:NE], InputGroup[:NE], p_ee_in)
        R_ee_group.connect(i=I, j=J)
        R_ee_group.w = w_ee_in + 0.1*w_ee_in*randn(len(R_ee_group.w))

    if p_ei_in > 0:
        R_ei_group = Synapses(Pe, Pi, model='w: 1', on_pre='g_ampa += w*gBarEx*nS')
        I, J = get_in_group_connectivity(InputGroup[:NE], InputGroup[NE:], p_ei_in)
        R_ei_group.connect(i=I, j=J)
        R_ei_group.w = w_ei_in + 0.1*w_ei_in*randn(len(R_ei_group.w))

    if p_ie_in > 0:
        R_ie_group = Synapses(Pi, Pe, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        I, J = get_in_group_connectivity(InputGroup[NE:], InputGroup[:NE], p_ie_in)
        R_ie_group.connect(i=I, j=J)
        R_ie_group.w = w_ie_in + 0.1*w_ie_in*randn(len(R_ie_group.w))

    if p_ii_in > 0:
        R_ii_group = Synapses(Pi, Pi, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        I, J = get_in_group_connectivity(InputGroup[NE:], InputGroup[NE:], p_ii_in)
        R_ii_group.connect(i=I, j=J)
        R_ii_group.w = w_ii_in + 0.1*w_ii_in*randn(len(R_ii_group.w))
    

   
    # ###########################################
    # Run Simulation without Plasticity
    # ###########################################

    # Spike Monitors
    sm = SpikeMonitor(P)
    signal_sp = SpikeMonitor(Signal)

    run(simtime)
    return signal_sp, sm
