import math
import random
import numpy as np
import itertools
import string
import scipy as sp
from toydown import GeoUnit, ToyDown
import argparse
import pickle


parser = argparse.ArgumentParser(description="Sample ToyDownMultiAttribute noise", 
                                 prog="run_toydown_noise.py")
parser.add_argument("n", metavar="num_samples", type=int,
                    help="How many runs to preform")
parser.add_argument("eps", metavar="epsilon", type=float,
                    help="total epsilon budget")
parser.add_argument("eps_splits", metavar="epsilon_splits", type=float, nargs='+',
                    help="list of how much epsilon to budget at each level")
args = parser.parse_args()


LEAVES = {'133': np.array([152.01127753, 138.84528446,  13.16599307]),
          '132': np.array([212.33036409, 161.9876171 ,  50.34274699]),
          '131': np.array([200.25783133, 120.10254376,  80.15528757]),
          '123': np.array([279.75309343, 116.00477968, 163.74831374]),
          '122': np.array([294.78230712, 100.31018182, 194.4721253 ]),
          '121': np.array([290.33488693, 112.70811816, 177.62676877]),
          '113': np.array([261.15279716, 102.36320046, 158.78959671]),
          '112': np.array([367.09928003, 101.20197765, 265.89730238]),
          '111': np.array([345.7709206 ,  60.39187144, 285.37904916])}

LEAF_NAMES = ["1" + "".join(a) for a in itertools.product(string.hexdigits[1:3+1], repeat=3-1)]

EPS = args.eps
EPS_SPLIT = args.eps_splits
N_SAMPS = args.n
NUM_LEAVES = len(LEAVES)
NUM_COLS = len(list(LEAVES.values())[0])

cons_0_diff = lambda n: [{'type': 'eq', 'fun': lambda x, i=i:  x[i] - np.sum([x[j] for j in range(i+1,i+3)])} 
                         for i in np.arange(n*3, step=3)]


def create_tree_from_leaves(leaf_dict):
    """ Given a dictionary, where the keys are the names of leaf nodes (labeled by their path)
        and the corresponding value is the associated attribute counts, this function returns
        the list of GeoUnits that defines the corresponding tree.
    """
    nodes = leaf_dict.copy()
    h = len(list(leaf_dict.keys())[0])
    n = len(list(leaf_dict.values())[0])
    
    for i in range(2, h+1):
        level_names = list(set(list(map(lambda s: s[:-(i-1)], leaf_dict.keys()))))
        level_counts = [np.zeros(n)]*len(level_names)
        for node in level_names:
            nodes[node] = np.array([v for k, v in leaf_dict.items() if k.startswith(node)]).sum(axis=0)
        
    return [GeoUnit(k, k[:-1], v) if k != "1" else GeoUnit(k, None, v) for k, v in nodes.items()]

def toydown_noise(leaves, model, cons=cons_0_diff, n_leaves=None):
    n = n_leaves if n_leaves else len(leaves)
    noised_counts = np.zeros((n, NUM_COLS))
    model.noise_and_adjust(node_cons=cons)
    for i,l in enumerate(leaves):
        noised_counts[i] = model.get_node(l).data.adjusted
    return noised_counts

print("Setting up model")
geounits = create_tree_from_leaves(LEAVES)
geounits.reverse()
model = ToyDown(geounits, 3, EPS, EPS_SPLIT)

print(args.eps)
print(args.eps_splits)
model.show()

print("Running Model")
noised_counts = np.zeros((N_SAMPS, NUM_LEAVES, NUM_COLS))

for i in range(N_SAMPS):
    noised_counts[i] = toydown_noise(LEAVES, model, n_leaves=NUM_LEAVES)
    if i % 100 == 0: print("*", end="", flush=True)

print()


print("Saving Results")

output = "data/ToyDownNoised_budget_{}_eps_{}_samps_{}.p".format(EPS, EPS_SPLIT, N_SAMPS)

with open(output, "wb") as f_out:
    pickle.dump(noised_counts, f_out)




