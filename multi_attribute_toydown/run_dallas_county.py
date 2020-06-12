import numpy as np
from toydown import GeoUnit, ToyDown
from dask.distributed import Client
import argparse
import multiprocessing

def create_tree_from_leaves(leaf_dict):
    """ Given a dictionary, where the keys are the names of leaf nodes (labeled by their path)
        and the corresponding value is the associated attribute counts, this function returns
        the list of GeoUnits that defines the corresponding tree.
    """
    nodes = leaf_dict.copy()
    h = len(list(leaf_dict.keys())[0])
    counts = ["TOTPOP", "VAP"]
    n = len(list(leaf_dict.values())[0][counts[0]])
    level_offsets = [3, 4, 10]
    
    for offset in level_offsets:
        level_names = list(set(list(map(lambda s: s[:-offset], leaf_dict.keys()))))
        for node in level_names:
            nodes[node] = {c: np.array([v[c] for k, v in leaf_dict.items() if k.startswith(node)]).sum(axis=0) for c in counts}
    return [GeoUnit(k, k[:-3], v) if len(k) == 15 else GeoUnit(k, k[:-1], v) if len(k) == 12 else GeoUnit(k, k[:-6], v) if len(k) == 11 else GeoUnit(k, k[:-3], v) for k, v in nodes.items()]

if __name__ == '__main__':
    multiprocessing.freeze_support()
    ## Set up args
    parser = argparse.ArgumentParser(description="ToyDownMultiAttribute noise Dallas County", 
                                     prog="run_dallas_county.py")
    parser.add_argument("w", metavar="num_workers", type=int,
                        help="How many cores to use")
    parser.add_argument("n", metavar="num_runs", type=int,
                        help="How many runs to preform")
    parser.add_argument("eps", metavar="epsilon", type=float,
                        help="total epsilon budget")
    parser.add_argument("eps_split", metavar="epsilon_split", type=str,
                        choices=["equal", "top_heavy", "mid_heavy", "bottom_heavy"],
                        help="how to budget epsilon across levels")
    args = parser.parse_args()


    ## Set up data
    print("Reading in Dallas County data")
    leaves_dallas_county = np.load("../data/dallas_county_blocks_all_pop_vap.p", allow_pickle=True)
    geounits_dc = create_tree_from_leaves(leaves_dallas_county)
    geounits_dc.reverse()

    counties = np.load("../data/texas_counties_all_pop_vap.p", allow_pickle=True)
    del counties["48113"] ## delete extra copy of Dallas County
    tx_data = {'TOTPOP': np.array([25145561,  9460921, 11397345,  2886825,    80586,   948426,
                                   17920,    33980,   319558]),
               'VAP': np.array([18279737,  6143144,  9074684,  2076282,    61856,   716968,
                                12912,    21205,   172686])}
    tx = GeoUnit("48", None, tx_data)
    county_geounits = [GeoUnit(geoid, "48", attr) for geoid, attr in counties.items()]
    county_geounits.insert(0, tx)


    split_dict = {"equal": [1/5, 1/5, 1/5, 1/5, 1/5], "top_heavy": [1/2, 1/4, 1/12, 1/12, 1/12], 
                  "mid_heavy": [1/12, 1/6, 1/2, 1/6, 1/12], "bottom_heavy": [1/12, 1/12, 1/12, 1/4, 1/2]}

    eps = args.eps
    eps_split_name = args.eps_split
    eps_split = split_dict[eps_split_name]
    n_samps = args.n
    height = len(eps_split)
    n_workers = args.w

    def run_model(i):
        return model_all.noise_and_adjust(verbose=False)

    client = Client(processes=True, threads_per_worker=1, n_workers=n_workers)
    print(client, flush=True)

    model_all = ToyDown(county_geounits + geounits_dc, height, eps, eps_split, gurobi=True)

    print("Starting {} runs with eps {} and {} split".format(n_samps*n_workers, eps, eps_split_name), flush=True)
    client.scatter(model_all, broadcast=True)

    for j in range(n_samps):
        adjusteds = client.map(run_model, range(n_workers))
        to_sav = np.array((client.gather(adjusteds)))
        np.save("test/dallas_county_2010_POP_VAP_eps_{}_{}_split_run_{}.npy".format(eps, eps_split_name, j), to_sav)
        del to_sav
        del adjusteds
        
    del model_all

