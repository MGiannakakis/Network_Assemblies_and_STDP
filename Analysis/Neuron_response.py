from brian2 import *


def get_tunings(We, Wi, NSigs = 8):
    
    its = 20
    Rates_b = np.zeros((its, NSigs))

    for i in range(its):
        for j in range(NSigs):
            Rates_b[i][j] = get_post_response_to_input(j, We, Wi, NSigs = NSigs)
            
    return Rates_b
    

def get_post_response_to_input(group, We, Wi, NSigs = 8):
    rate = 50
    pulse_group = group

    noise_ratio = 0.999
    simtime = 1 * second  # Simulation time

    # #############
    # Neuron model
    # #############

    tau_ampa = 5.0 * ms  # Glutamatergic synaptic time constant
    tau_gaba = 10.0 * ms  # GABAergic synaptic time constant
    tau_alpha = 80.0 * ms

    tau_stdp = 10 * ms  # STDP time constant

    gl = 10.0 * nsiemens  # Leak conductance
    el = -60 * mV  # Resting potential
    er = -80 * mV  # Inhibitory reversal potential
    memc = 200.0 * pfarad  # Membrane capacitance

    gBarEx = 0.14
    gBarIn = 0.35

    eq = '''
    dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er)))/memc : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gaba/dt = -g_gaba/tau_gaba : siemens
    '''

    #########################################
    # Input
    #########################################

    NCells = 1000  # Number of input spike trains.
    ExFrac = 0.8  # Fraction of Excitatory spike trains.
    # Number of spike trains in each excitatory input group
    ExGroupsize = round((NCells * ExFrac) / NSigs)

    # Number of spike trains in each inhibitory input group
    InGroupsize = round((NCells * (1 - ExFrac)) / NSigs)

    # ###########################################
    # Initialize neuron
    # ###########################################

    N = NeuronGroup(1, model=eq, threshold='v > vt', reset='v=el', refractory=5 * ms, method='euler')
    N.v = el

    ###################################
    # Input Groups
    ##################################

    temptype = -1
    InputGroup = np.zeros(NCells)

    for ii in range(NCells):
        if (ii <= NCells * ExFrac):
            if ((ii - 1) % ExGroupsize == 0):
                temptype += 1
            InputGroup[ii] = temptype
        else:
            if (ii % InGroupsize == 0):
                temptype -= 1
            InputGroup[ii] = temptype

    InputGroup[0] = InputGroup[1]

    #############################
    # Set the Rates
    # ###########################

    def get_noise_pulse(InputGroup, noise_ratio, rate, pulse_group):
        noise_rates = []
        for i in range(len(InputGroup)):
            if InputGroup[i] == pulse_group:
                noise_rates.append(noise_ratio * rate)
            else:
                noise_rates.append(0)
        return noise_rates * Hz

    Noise_rates = get_noise_pulse(InputGroup, noise_ratio, rate, pulse_group)

    s_rates = np.zeros(NSigs)
    s_rates[pulse_group] = (1 - noise_ratio) * rate * Hz
    Signal_rates = s_rates * Hz

    # ###########################################
    # Create Incoming Spike Trains
    # ###########################################

    Signal = PoissonGroup(NSigs, Signal_rates)
    Noise = PoissonGroup(NCells, Noise_rates)

    vr = -60 * mV  # Resting Potential
    vt = -50 * mV  # Theshold
    tau = 10 * ms

    eqR = '''
    dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er)))/memc : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gaba/dt = -g_gaba/tau_gaba : siemens
    sumw :1
    '''

    P = NeuronGroup(NCells, model=eqR, threshold='v > vt', reset='v = el', refractory=5 * ms, method='euler')

    ################################
    # Signal feedforward connections
    ################################
    w_input = 35

    S = Synapses(Signal, P, on_pre='g_ampa += w_input*gBarEx*nS')

    S.connect(i=InputGroup.astype(int), j=range(NCells))

    Noise_Synapses = Synapses(Noise, P, on_pre='g_ampa += w_input*gBarEx*nS')
    Noise_Synapses.connect(condition='i==j')

    # Neural Populations
    NE = int(NCells * ExFrac)
    NI = NCells - NE
    Pe = P[:NE]
    Pi = P[NE:]        

    #################
    # Connections
    #################

    con_e = Synapses(Pe, N, 'w: 1', on_pre='g_ampa += gBarEx*w*nS')
    con_e.connect()
    con_e.w = We

    con_ie = Synapses(Pi, N, 'w: 1', on_pre='g_gaba += gBarIn*w*nS')
    con_ie.connect()
    con_ie.w = Wi

    # Spike Monitors
    M = SpikeMonitor(N)

    ###########################################
    # Run Simulation without Plasticity
    # ###########################################
    run(simtime)

    # Post FR
    s_post = M.spike_trains()[0]
    fr = round(len(s_post / second) / (simtime / second), 3)
    
    return fr











