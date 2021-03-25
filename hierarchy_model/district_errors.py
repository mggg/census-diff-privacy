from randomDistricts import Hierarchy_2D
from gerrychain import (Graph, Partition, MarkovChain, tree,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.updaters import Tally, cut_edges
from gerrychain.proposals import recom
from treelib import Node, Tree
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from functools import partial
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange

NUM_DISTS = 1000
NUM_RUNS = 16

SPLITS = {"equal": [0.2,0.2,0.2,0.2,0.2], 
          "top_heavy": [1/2, 1/4, 1/12, 1/12, 1/12],
          "mid_heavy": [1/12, 1/6, 1/2, 1/6, 1/12], 
          "bottom_heavy": [1/12, 1/12, 1/12, 1/4, 1/2],
          "bg_heavy": [1/12, 1/6, 1/6, 1/2, 1/12]}

POPCOLS = ['HISP', 'NH_WHITE', 'NH_BLACK', 'NH_AMIN', 
            'NH_ASIAN', 'NH_NHPI', 'NH_OTHER*']

# tract_recom_ds = np.load("sample_districts/1000_recom_tract_dists.npy", allow_pickle=True)
# bg_recom_ds = np.load("sample_districts/1000_recom_bg_dists.npy", allow_pickle=True)
# pct_recom_ds = np.load("sample_districts/1000_recom_pct_dists.npy", allow_pickle=True)
# block_recom_ds = np.load("sample_districts/1000_recom_block_dists.npy", allow_pickle=True)

# block_discon_ds = np.load("sample_districts/400_discon_blocks.npy", allow_pickle=True)
# tract_discon_ds = np.load("sample_districts/400_discon_tracts.npy", allow_pickle=True)
# block_bb = np.load("sample_districts/400_squre_block_dists.npy", allow_pickle=True)[0]["dicts"]


# tract_recom_4 = np.load("sample_districts/tract_parts_debug/1000_recom_tract_dists.npy", 
#                             allow_pickle=True)
# tract_discon_4 = np.load("sample_districts/tract_parts_debug/1000_discon_tract_dists.npy", 
#                              allow_pickle=True)

# tract_recom_10 = np.load("sample_districts/tract_parts_debug/1000_recom_tract_dists_10.npy", 
#                             allow_pickle=True)
# tract_discon_10 = np.load("sample_districts/tract_parts_debug/1000_discon_tract_dists_10.npy", 
#                              allow_pickle=True)

# tract_recom_20 = np.load("sample_districts/tract_parts_debug/1000_recom_tract_dists_20.npy", 
#                             allow_pickle=True)
# tract_discon_20 = np.load("sample_districts/tract_parts_debug/1000_discon_tract_dists_20.npy", 
#                              allow_pickle=True)

# block_recom_4 = np.load("sample_districts/1000_recom_block_dists.npy", allow_pickle=True)


# block_recom_10 = np.load("sample_districts/tract_parts_debug/1000_recom_block_dists_10.npy", 
#                             allow_pickle=True)
# block_discon_10 = np.load("sample_districts/tract_parts_debug/1000_discon_block_dists_10.npy", 
#                              allow_pickle=True)

# block_recom_20 = np.load("sample_districts/tract_parts_debug/1000_recom_block_dists_20.npy", 
#                             allow_pickle=True)
# block_discon_20 = np.load("sample_districts/tract_parts_debug/1000_discon_block_dists_20.npy", 
#                              allow_pickle=True)

block_recom_175 = np.load("sample_districts/tract_parts_debug/1000_recom_block_dists_175.npy", 
                            allow_pickle=True)
block_discon_175 = np.load("sample_districts/tract_parts_debug/1000_discon_block_dists_175.npy", 
                             allow_pickle=True)

# block_recom_10_fs = np.load("sample_districts/tract_parts_debug/1000_recom_block_frags_10.npy", 
#                             allow_pickle=True)
# block_discon_10_fs = np.load("sample_districts/tract_parts_debug/1000_discon_block_frags_10.npy", 
#                              allow_pickle=True)

# block_recom_20_fs = np.load("sample_districts/tract_parts_debug/1000_recom_block_frags_20.npy", 
#                             allow_pickle=True)
# block_discon_20_fs = np.load("sample_districts/tract_parts_debug/1000_discon_block_frags_20.npy", 
#                              allow_pickle=True)

# block_recom_175_fs = np.load("sample_districts/tract_parts_debug/1000_recom_block_frags_175.npy", 
#                             allow_pickle=True)
# block_discon_175_fs = np.load("sample_districts/tract_parts_debug/1000_discon_block_frags_175.npy", 
#                              allow_pickle=True)

bg_recom_175 = np.load("sample_districts/tract_parts_debug/1000_recom_bg_dists_175_eps_0.25.npy", 
                            allow_pickle=True)
bg_discon_175 = np.load("sample_districts/tract_parts_debug/1000_discon_bg_dists_175.npy", 
                             allow_pickle=True)

with open("data/dallas_hierarcy.p", "rb") as fin:
    h = pickle.load(fin)
dallas_blks = h.gdf#.set_index("GEOID10")
# dallas_blk_recon = pd.read_csv("../data/dallas_reconstruction.csv")
topdown_run_cols = ["Run_{}".format(i+1) for i in range(NUM_RUNS)]

dist_blks = lambda d: [k for k,v in d.items() if v]

def unique_dists(ds):
    d_blks = np.array(list(map(dist_blks, ds)))
    return np.unique(d_blks)[:NUM_DISTS]

for alg, ds, frags in [ 
                #  ("block_discon_10", block_discon_10, block_discon_10_fs), 
                #  ("block_recom_10", block_recom_10, block_recom_10_fs),
                #  ("block_discon_20", block_discon_20, block_discon_20_fs), 
                #  ("block_recom_20", block_recom_20, block_recom_20_fs),
                #  ("block_discon_175", block_discon_175, np.ones(NUM_DISTS)), 
                #  ("block_recom_175", block_recom_175,  np.ones(NUM_DISTS)),
                #  ("tract_discon_4", tract_discon_4, np.ones(NUM_DISTS)), 
                #  ("tract_recom_4", tract_recom_4, np.ones(NUM_DISTS)),
                #  ("tract_discon_10", tract_discon_10, np.ones(NUM_DISTS)), 
                #  ("tract_recom_10", tract_recom_10, np.ones(NUM_DISTS)),
                #  ("tract_discon_20", tract_discon_20, np.ones(NUM_DISTS)), 
                #  ("tract_recom_20", tract_recom_20, np.ones(NUM_DISTS)),
                # ("tract_recom", tract_recom_ds), ("blockgroup_recom", bg_recom_ds), 
                # ("precinct_recom", pct_recom_ds), ("block_recom", block_recom_ds),
                # ("block_discon", block_discon_ds), ("tract_discon", tract_discon_ds),
                # ("block_bounding_box", block_bb)
                ("bg_discon_175", bg_discon_175, np.ones(NUM_DISTS)), 
                ("bg_recom_175", bg_recom_175,  np.ones(NUM_DISTS)),
                ]:
                
    print("Starting {} calculations".format(alg), flush=True)
    splits = ["equal", "mid_heavy", "bottom_heavy", "bg_heavy"]#, "top_heavy", "mid_heavy", "bottom_heavy", "bg_heavy"] #if (alg.endswith("discon") or alg == "block_recom") else []
    d_blks = np.array(list(map(dist_blks, ds))) #unique_dists(ds)
    for split in splits:
        print("Split: {}".format(split))
        toydown_allow_neg = pd.read_csv("../multi_attribute_toydown/results/new_recon_no_POP_VAP_runs_{}_1_{}.csv".format("allow_neg", split))
        toydown_non_neg = pd.read_csv("../multi_attribute_toydown/results/new_recon_no_POP_VAP_runs_{}_1_{}.csv".format("non_neg", split))
        toydown_allow_neg["TOTPOP"] = toydown_allow_neg[POPCOLS].sum(axis=1)
        toydown_non_neg["TOTPOP"] = toydown_non_neg[POPCOLS].sum(axis=1)
        toydown_allow_neg["GEOID"] = toydown_allow_neg["GEOID"].astype(str)
        toydown_non_neg["GEOID"] = toydown_non_neg["GEOID"].astype(str)
        toydown_allow_neg.set_index("GEOID", inplace=True)
        toydown_non_neg.set_index("GEOID", inplace=True)

        toydown_non_neg_no_empty = pd.read_csv("../multi_attribute_toydown/results/new_recon_no_POP_VAP_runs_non_empty_blocks_{}_1_{}.csv".format("non_neg", split))
        toydown_non_neg_no_empty["TOTPOP"] = toydown_non_neg_no_empty[POPCOLS].sum(axis=1)
        toydown_non_neg_no_empty.GEOID = toydown_non_neg_no_empty.GEOID.astype(str)
        toydown_non_neg_no_empty = pd.merge(dallas_blks, toydown_non_neg_no_empty, left_on="GEOID10", 
                                            right_on="GEOID", how="left").fillna(0)
        toydown_non_neg_no_empty.set_index("GEOID10", inplace=True)

        topdown_no_hh = pd.read_csv("../tot_pops/TEXAS_STUB_{}_1_block_pops.csv".format(split))
        topdown_no_hh.GEOID10 = topdown_no_hh.GEOID10.astype(str)
        topdown_no_hh = pd.merge(dallas_blks, topdown_no_hh, left_on="GEOID10", right_on="GEOID10", how="left").fillna(0)
        topdown_no_hh.set_index("GEOID10", inplace=True)

        df = pd.DataFrame(columns=["dist_id", "frag_score", "TOTPOP10", "model", "Run", "TOTPOP_N"])

        for i in tqdm(range(NUM_DISTS)):
            d = d_blks[i]
            district_type = alg
            district_id = alg + "_{}".format(i)
            # frag_score = h.assign_district_tree_variance(d, eps=1, sensitivity=1,
            #                                              eps_splits=[np.sqrt(2)]*5)
            frag_score = frags[i]
            totpop = dallas_blks.set_index("GEOID10").loc[d].TOTPOP10.sum()
            toydown_allow_neg_pops = toydown_allow_neg.loc[d].groupby("run").sum().TOTPOP.values
            toydown_non_neg_pops = toydown_non_neg.loc[d].groupby("run").sum().TOTPOP.values
            toydown_non_neg_no_empty_pops = toydown_non_neg_no_empty.loc[d].groupby("run").sum().TOTPOP.values
            topdown_pops = topdown_no_hh.loc[d][topdown_run_cols].sum(axis=0).values

            models = [("ToyDown_non_empty_non_neg", toydown_non_neg_no_empty_pops),
                      ("ToyDown_allow_neg", toydown_allow_neg_pops), 
                      ("ToyDown_non_neg", toydown_non_neg_pops), 
                      ("TopDown_no_HH_cons", topdown_pops)
                      ]
            for j, (m_name, m_pops) in enumerate(models):
                for k in range(NUM_RUNS):
                    idx = i*len(models)*NUM_RUNS + j*NUM_RUNS + k
                    df.loc[idx] = [district_id, frag_score, totpop, m_name, k, m_pops[k]]


        df = df.assign(split=split, district_type=alg)
        df = df.assign(Error=lambda r: r["TOTPOP_N"] - r["TOTPOP10"],
                  ErrorMag=lambda r: np.abs(r["Error"]))
        df.to_csv("dist_errors/{}_{}_split_errors.csv".format(alg, split), index=False)