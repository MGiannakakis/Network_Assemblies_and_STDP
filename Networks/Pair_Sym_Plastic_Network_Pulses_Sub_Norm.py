from brian2 import *
from Connectivity_Utils import *

def Network_get_Stats(W = 0.0,
                p = 0.0,
                noise_ratio = 0.01,
                R = [1., 1., 1., 1.]):

    # #############
    # Neuron model
    # #############
    target_rate = 3
    NSigs = 8
    Assemblies = True

    #Clustering Parameters
    r_ee = R[0] * 1 / (NSigs - 1)
    r_ei = R[1] * 1 / (NSigs - 1)
    r_ie = R[2] * 1 / (NSigs - 1)
    r_ii = R[3] * 1 / (NSigs - 1)

    tau_ampa = 5.0 * ms  # Glutamatergic synaptic time constant
    tau_gaba = 10.0 * ms  # GABAergic synaptic time constant

    tau_stdp = 10 * ms  # STDP time constant

    learningtime = 100 * second  # Simulation time
    pulsetime = 1 * second

    gl = 10.0 * nsiemens  # Leak conductance
    el = -60 * mV  # Resting potential
    er = -80 * mV  # Inhibitory reversal potential
    vt = -50 * mV  # Theshold
    memc = 200.0 * pfarad  # Membrane capacitance
    bgcurrent = 200 * pA  # External current

    gBarEx = 0.14
    gBarIn = 0.35

    eq = '''
    dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er))+bgcurrent)/memc : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gaba/dt = -g_gaba/tau_gaba : siemens
    sumwe :1
    sumwi :1
    '''


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

    # ###########################################
    # Initialize neuron
    # ###########################################

    N = NeuronGroup(1, model=eq, threshold='v > vt',
                    reset='v=el', refractory=5 * ms, method='euler')
    N.v = el

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

    noise_rate = noise_ratio*50
    active_rate = (1-noise_ratio)*100
    background_rate = (1-noise_ratio)*25
    activation_prob =  0.2*ones(NSigs)    
    Gens = int(learningtime/pulsetime)
    sigs = np.random.randint(0, NSigs, Gens)

    stimulus = TimedArray(np.stack([get_rates(NSigs, g, active_rate, background_rate, activation_prob).tolist()
                                         for g in sigs])*Hz, dt=pulsetime)

    Signal = PoissonGroup(NSigs, rates='stimulus(t, i)')
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
        R_ee.connect(p=p_ee_out)
        R_ee.w = w_ee_out + 0.1*w_ee_out*randn(len(R_ee.w))

    if p_ei_out > 0:
        R_ei = Synapses(Pe, Pi, model='w: 1', on_pre='g_ampa += w*gBarEx*nS')
        R_ei.connect(p=p_ei_out)
        R_ei.w = w_ei_out + 0.1*w_ei_out*randn(len(R_ei.w))

    if p_ie_out > 0:
        R_ie = Synapses(Pi, Pe, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        R_ie.connect(p=p_ie_out)
        R_ie.w = w_ie_out + 0.1*w_ie_out*randn(len(R_ie.w))

    if p_ii_out > 0:
        R_ii = Synapses(Pi, Pi, model='w: 1', on_pre='g_gaba += i_s*w*gBarIn*nS')
        R_ii.connect(p=p_ii_out)
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

    #############################
    # Excitatory Plasticity
    ############################

    A_LTD = 0.1
    A_LTP = 1.0
    
    w_target = 5.0
    gmax = 100  # Maximum  weight
    
    # Equations
    eqs_stdp_exc = '''
    w : 1 
    sumwe_post = w   : 1   (summed)
    dApre/dt=-Apre/tau_stdp : 1 (event-driven)
    dApost/dt=-Apost/tau_stdp : 1 (event-driven)
    '''

    # equations executed only when a presynaptic spike occurs
    Pre_eq = '''
             g_ampa += gBarEx*w*nS
             Apre += 1.
             w_minus = eta_e*A_LTD*Apost 
             w = clip(w - w_minus, 0.001, gmax)
             w += eta_norm*w*(NE*w_target - sumwe_post)/NE
             '''
    # equations executed only when a postsynaptic spike occurs
    Post_eq = '''
              Apost += 1.                                                                                        
              w_plus = eta_e*A_LTP*Apre
              w = clip(w + w_plus, 0.001, gmax)
             w += eta_norm*w*(NE*w_target - sumwe_post)/NE
              '''
    con_ee = Synapses(Pe, N, model=eqs_stdp_exc,
                      on_pre=Pre_eq,
                      on_post=Post_eq,
                      delay=0*ms)
    con_ee.connect()
    con_ee.w = 0.1

    #############################
    # Inhibitory Plasticity
    ############################

    eqs_stdp_inhib = '''
    w : 1
    sumwi_post = w   : 1   (summed)
    dApre/dt=-Apre/tau_stdp : 1 (event-driven)
    dApost/dt=-Apost/tau_stdp : 1 (event-driven)
    '''
    alpha = 3 * Hz * tau_stdp * 2  # Target rate parameter

    c_i = gBarIn/gBarEx
    con_ie = Synapses(Pi, N, model=eqs_stdp_inhib,
                      on_pre='''Apre += 1.
                             w = clip(w + (Apost - alpha)*eta_i, 0.001, gmax)
                             g_gaba += gBarIn*w*nS''',
                      on_post='''Apost += 1.
                              w = clip(w + Apre*eta_i, 0.001, gmax)
                           ''',
                      delay=0*ms)
    con_ie.connect()
    con_ie.w = 0.1    
    
    # ###########################################
    # Run Simulation with Plasticity
    # ###########################################
    eta_e = 2.5*1e-3  #E Learning rate
    eta_i = 1e-2  #1e-2 #I Learning rate
    eta_norm = 3*1e-3 
    
    # Spike Monitors
    sm = SpikeMonitor(P)
    post_spikes = SpikeMonitor(N)
    signal_sp = SpikeMonitor(Signal)
    
     #State Monitor
    mon = StateMonitor(N, variables = ['v', 'g_ampa', 'g_gaba'], record=True)
   
    run(learningtime)
    
    
    return [signal_sp, sm, post_spikes, con_ee.w, con_ie.w, mon]

