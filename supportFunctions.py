import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def print_stepcount():
    with open(f"steplist.pickle", "rb") as handle:
        steplist = pickle.load(handle)
    
    plt.scatter(range(len(steplist)), steplist, s= 10)
    plt.show()


print_stepcount()
