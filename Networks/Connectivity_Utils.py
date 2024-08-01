from brian2 import *

def get_rates(NSigs, index, active_rate, background_rate, activation_prob):
    rates = np.zeros(NSigs)
    for s in range(NSigs):
        if rand() < activation_prob[s]:
            rates[s] = active_rate
        else:
             rates[s] = background_rate
    return rates

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
    return Out, In


def get_recurrent_connectivity(pre, post, prob_in, prob_out, w_in, w_out, w_noise = 1e-1):
    Out = []
    In = []
    W = []
    for i in range(len(pre)):
        for j in range(len(post)):
            if pre[i] == post[j]:
                if rand() < prob_in:
                    Out.append(i)
                    In.append(j)
                    W.append(w_in + w_noise*w_in*randn())
            else:
                if rand() < prob_out:
                    Out.append(i)
                    In.append(j) 
                    W.append(w_out + w_noise*w_out*randn())
    return Out, In, W


def get_con_params_p(NSigs, r, V):
    d = (NSigs - 1) * (r + 1)

    v_in = (NSigs * r - r - 1) * NSigs * V / d
    v_out = NSigs * V / d

    return v_in, v_out

def get_con_params(NSigs, r, V):
    d = (NSigs - 1) * (r + 1)

    v_in = r * (NSigs - 1) * NSigs * V / d
    v_out = NSigs * V / d

    return v_in, v_out


