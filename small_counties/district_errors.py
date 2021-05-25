# from gerrychain import (Graph, Partition, MarkovChain, tree,
#                         proposals, updaters, constraints, accept, Election)
# from gerrychain.updaters import Tally, cut_edges
# from gerrychain.proposals import recom
from treelib import Node, Tree
import geopandas as gpd
import numpy as np
# import matplotlib.pyplot as plt
import pickle
from functools import partial
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
# from local_tools.randomDistricts import Hierarchy_2D

NUM_DISTS = 1000
NUM_RUNS = 16

SPLITS = {"equal": [0.2,0.2,0.2,0.2,0.2], 
          "top_heavy": [1/2, 1/4, 1/12, 1/12, 1/12],
          "mid_heavy": [1/12, 1/6, 1/2, 1/6, 1/12], 
          "bottom_heavy": [1/12, 1/12, 1/12, 1/4, 1/2],
          "bg_heavy": [1/12, 1/6, 1/6, 1/2, 1/12]}

POPCOLS = ['HISP', 'NH_WHITE', 'NH_BLACK', 'NH_AMIN', 
            'NH_ASIAN', 'NH_NHPI', 'NH_OTHER*']

topdown_run_cols = ["Run_{}".format(i+1) for i in range(NUM_RUNS)]
dist_blks = lambda d: [k for k,v in d.items() if v]

def unique_dists(ds):
    d_blks = np.array(list(map(dist_blks, ds)))
    return np.unique(d_blks)[:NUM_DISTS]

def dist_errors(alg, ds, blocks, frags=np.ones(NUM_DISTS)):
    print("Starting {} calculations".format(alg), flush=True)
    splits = ["equal", "mid_heavy", "bottom_heavy", "bg_heavy"]#, "top_heavy", "mid_heavy", "bottom_heavy", "bg_heavy"] #if (alg.endswith("discon") or alg == "block_recom") else []
    d_blks = np.array(list(map(dist_blks, ds))) #unique_dists(ds)
    for split in splits:
        print("Split: {}".format(split))
        toydown_allow_neg = pd.read_csv("../multi_attribute_toydown/results/small_cntys/ToyDown_runs_allow_neg_1_{}.csv".format(split))
        toydown_non_neg = pd.read_csv("../multi_attribute_toydown/results/small_cntys/ToyDown_runs_non_neg_1_{}.csv".format(split))
        toydown_allow_neg["TOTPOP"] = toydown_allow_neg[POPCOLS].sum(axis=1)
        toydown_non_neg["TOTPOP"] = toydown_non_neg[POPCOLS].sum(axis=1)
        toydown_allow_neg["GEOID"] = toydown_allow_neg["GEOID"].astype(str)
        toydown_non_neg["GEOID"] = toydown_non_neg["GEOID"].astype(str)
        toydown_allow_neg.set_index("GEOID", inplace=True)
        toydown_non_neg.set_index("GEOID", inplace=True)

        # topdown_no_hh = pd.read_csv("../tot_pops/TEXAS_STUB_{}_1_block_pops.csv".format(split))
        # topdown_no_hh.GEOID10 = topdown_no_hh.GEOID10.astype(str)
        # topdown_no_hh = pd.merge(blocks, topdown_no_hh, left_on="GEOID10", right_on="GEOID10", how="left").fillna(0)
        # topdown_no_hh.set_index("GEOID10", inplace=True)

        df = pd.DataFrame(columns=["dist_id", "frag_score", "TOTPOP10", "model", "Run", "TOTPOP_N"])

        for i in tqdm(range(NUM_DISTS)):
            d = d_blks[i]
            district_id = alg + "_{}".format(i)
            # frag_score = h.assign_district_tree_variance(d, eps=1, sensitivity=1,
            #                                              eps_splits=[np.sqrt(2)]*5)
            frag_score = frags[i]
            totpop = blocks.set_index("GEOID10").loc[d].TOTPOP.sum()
            toydown_allow_neg_pops = toydown_allow_neg.loc[d].groupby("run").sum().TOTPOP.values
            toydown_non_neg_pops = toydown_non_neg.loc[d].groupby("run").sum().TOTPOP.values
            # topdown_pops = topdown_no_hh.loc[d][topdown_run_cols].sum(axis=0).values

            models = [("ToyDown_allow_neg", toydown_allow_neg_pops), 
                      ("ToyDown_non_neg", toydown_non_neg_pops), 
                    #   ("TopDown_no_HH_cons", topdown_pops)
                      ]
            for j, (m_name, m_pops) in enumerate(models):
                for k in range(NUM_RUNS):
                    idx = i*len(models)*NUM_RUNS + j*NUM_RUNS + k
                    df.loc[idx] = [district_id, frag_score, totpop, m_name, k, m_pops[k]]


        df = df.assign(split=split, district_type=alg)
        df = df.assign(Error=lambda r: r["TOTPOP_N"] - r["TOTPOP10"],
                  ErrorMag=lambda r: np.abs(r["Error"]))
        df.to_csv("dist_errors/{}_{}_split_errors.csv".format(alg, split), index=False)

"""
    Galveston City Block ReCom
"""
galv_city_blocks = gpd.read_file("shapes/blocks/galveston_city_blocks_2010_data.shp")
block_recom = np.load("sample_districts/galveston_city_1000_recom_block_dists_6_eps_0.02.npy", allow_pickle=True)

dist_errors("galveston_city_block_recom", block_recom, galv_city_blocks)

"""
    5 Counties (Bell, Brazoria, Cameron, Galveston, and Nueces) Tract ReCom/Discon
"""
counties = ["bell", "brazoria","cameron","galveston","nueces"]
for county in counties:
    blocks = gpd.read_file("shapes/blocks/{}_county_blocks_2010_data.shp".format(county))
    tract_recom = np.load("sample_districts/{}_county_1000_recom_tract_dists_4_eps_0.02.npy".format(county), allow_pickle=True)
    tract_discon = np.load("sample_districts/{}_county_1000_discon_tract_dists_4_eps_0.02.npy".format(county), allow_pickle=True)

    dist_errors("{}_county_tract_recom".format(county), tract_recom, blocks)
    dist_errors("{}_county_tract_discon".format(county), tract_discon, blocks)