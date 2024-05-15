""" Run Checks for different levels of connection probability

"""
from sbi.simulators.simutils import simulate_in_batches
from functools import partial
import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from scipy.io import loadmat as loadmat
na =np.array
import torch 
from Balance_Network import Network_get_Tuning as NetworkRunPlasticity
from Balance_Network import get_con_params_p 

def clear_brian2():
    try:
        clear_cache('cython')
    except:
        pass
    return None


#load the large grid

def simulator(params):
    #set a seed
    np.random.seed(int(params[6].numpy()))
    #parse paramters
    R = params[:4].numpy()
    W = params[4].numpy()
    p = float(params[7].numpy())
    noise_ratio = params[5].numpy()
    try:
        output =  NetworkRunPlasticity(W=W,p=p,noise_ratio=noise_ratio,R=R, Assemblies=False,get_con_params=get_con_params_w,)
        
    except:
        print('here')
        raise Exception(params)
    return torch.tensor(output)

## Across rows, i.e. different noises,fixed W
# Get the parameters of interest
n_runs = 5 # repear n times with different seeds 
master_seed = 12345679

ps = [0.1,0.25,0.5,0.95]
theta_row = np.zeros(shape=(len(ps)*n_runs,8))
seeds = np.arange(master_seed,master_seed+n_runs,1)
na = np.array
# params = [na([ 8.8354248 , 10.1017701 ,  2.36454469,  0.98448946,  1.00797251]),
#  na([11.43642553,  7.19067017,  3.52128833,  1.45457922,  8.68030447]),
#  na([9.75732195, 6.41504679, 3.84382069, 2.28119234, 9.27967888]),
#  na([9.6155116 , 9.9445414 , 4.12120374, 4.36610407, 8.91170345])]


# params = [na([7.58253231, 6.22559621, 2.40019505, 1.67144261, 1.0716284 ]),
#  na([ 8.83545735, 10.1017974 ,  2.36453747,  0.98449337,  1.00797365]),
#  na([11.43642837,  7.19069868,  3.52128829,  1.45458651,  8.68028708]),
#  na([9.75734067, 6.41511136, 3.84384328, 2.28123503, 9.27969188]),
#  na([8.6022762 , 8.3346886 , 4.14012841, 4.10180404, 6.60914918])]
params = [array([9.62421209, 9.03345573, 2.79774428, 0.97764214]),
 array([10.84984187,  6.82522573,  3.79135385,  2.20275562]),
 array([11.24195144,  8.39948314,  4.60174421,  3.49450984]),
 array([10.42163123,  9.92706761,  4.82735694,  4.75511952]),
 array([10.9436816 , 10.38549364,  5.84189028,  5.64141224])]

ps = [0.1,0.25,0.5,0.75,0.95]
# params = [na([9.62417645, 9.03343165, 2.7977286 , 0.97763822])]

n = 0
for p_i,p in enumerate(ps):
    for seed in seeds:
        theta_row[n,:4] =params[p_i][:4]
        theta_row[n,4]= 1/p#params[p_i][4]
        theta_row[n,5]=0.2#0.25
        theta_row[n,6]=seed
        theta_row[n,7]=p
        n=n+1
    
theta_row_null = np.zeros(shape=(len(ps)*n_runs,8))
n= 0
for p_i,p in enumerate(ps):
    for seed in seeds:
        theta_row_null[n,:4] =[1.,1.,1.,1.]
        theta_row_null[n,4]= 1/p#params[p_i][4]
        theta_row_null[n,5]=0.2#0.25
        theta_row_null[n,6]=seed
        theta_row_null[n,7]=p
        n=n+1


db_collected = []
db_null_collected = []
def batch_simulatior(theta: torch.Tensor): 
    """Return a batch of simulations by looping over a batch of parameters."""
    assert theta.ndim > 1, "Theta must have a batch dimension."
    xs = list(map(simulator, theta))
    return torch.cat(xs, dim=0).reshape(theta.shape[0], -1)
bd = simulate_in_batches(batch_simulatior,torch.tensor(theta_row),sim_batch_size=1, num_workers=20)
clear_brian2()
bd_null = simulate_in_batches(batch_simulatior,torch.tensor(theta_row_null),sim_batch_size=1, num_workers=20)
clear_brian2()
db_collected.append(bd)
db_null_collected.append(bd_null)
np.save('results/DB_ps_full_v3_s2n=0.2_woverp.npy',[db_collected,db_null_collected]) #


