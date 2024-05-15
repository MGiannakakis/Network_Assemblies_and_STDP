# ABC over different sparsities 

# %matplotlib inline
import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import matplotlib.pyplot as plt
# from sbi.inference import SNPE, SNRE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils_sbi
from ABC import *


for p in [0.95]:
    # if p==0.95:
        # cut_g= True
    # else:
    cut_g= True
    model = MyModel([1,0,0,0],tmax=3,n_sim=60,p=p,simulator =simulatorW)
    model.set_prior([stats.uniform(1,10),
                     stats.uniform(1,10),
                    stats.uniform(1,10),
                    stats.uniform(1,10)])
    posterior2 = pmc_abc(model,[1,0,0,0],
                        epsilon_0=.5, 
                        min_samples=120, 
                        steps=22,
                        file='results/ABC_p=%s_v1_4par_n=2000'%p,
                        minError=.001,
                        cut_g= cut_g)





