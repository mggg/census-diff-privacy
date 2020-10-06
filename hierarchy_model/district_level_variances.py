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


h = Hierarchy_2D("../../data/dallas_county_blocks/dallas_county_blocks10.shp", "GEOID10", "TOTPOP10")

epsilon_split = {"equal": [0.2,0.2,0.2,0.2,0.2], 
                 "top_heavy": [1/2, 1/4, 1/12, 1/12, 1/12],
                 "mid_heavy": [1/12, 1/6, 1/2, 1/6, 1/12], 
                 "bottom_heavy": [1/12, 1/12, 1/12, 1/4, 1/2],
                 "bg_heavy": [1/12, 1/6, 1/6, 1/2, 1/12]}

pop_cols = ['HISP', 'NH_WHITE', 'NH_BLACK', 'NH_AMIN', 
             'NH_ASIAN', 'NH_NHPI', 'NH_OTHER*']


def district_frag_scores(h, districts):
    num_districts = len(districts)
    frag_scores = np.zeros(num_districts)
    for i in tqdm(range(num_districts)):
        frag_scores[i] = h.assign_district_tree_variance(districts[i], eps=1, 
                                                         eps_splits=[np.sqrt(2)]*5,
                                                         sensitivity=1)
    return frag_scores


def toydown_empirical_variances(districts, neg_flag="non_neg"):
    vrs = {}
    num_districts = len(districts)
    for split in ["equal", "top_heavy", "mid_heavy", "bottom_heavy", "bg_heavy"]:
        df = pd.read_csv("../multi_attribute_toydown/results/new_recon_no_POP_VAP_runs_{}_1_{}.csv".format(neg_flag, split))
        df["TOTPOP"] = df[pop_cols].sum(axis=1)
        pops = np.zeros((num_districts, 16))
        print(split)
        for j in tqdm(range(num_districts)):
            pops[j] = df[df.GEOID.astype(str).apply(lambda i: bool(districts[j][i]))].groupby("run").sum().TOTPOP.values
        print() 
        vs = np.var(pops, axis=1)
        vrs[split] = vs
    return vrs


def topdown_empirical_variances(districts, household_cons=False):
    vrs = {}
    num_districts = len(districts)
    run_cols = ["Run_{}".format(i+1) for i in range(20)]
    dallas_blk_recon = pd.read_csv("../data/dallas_reconstruction.csv")
    splits = ["equal", "top_heavy", "mid_heavy", "bottom_heavy"]
    if not household_cons:
        splits.append("bg_heavy")
    for split in splits:
        file = "../tot_pops/TEXAS_STUB_HH_{}_1_block_pops.csv".format(split) if household_cons else "../tot_pops/TEXAS_STUB_{}_1_block_pops.csv".format(split)
        df = pd.read_csv(file)
        df = pd.merge(dallas_blk_recon, df, left_on="geoid", right_on="GEOID10", how="left").fillna(0)
        pops = np.zeros((num_districts, 20))
        print(split)
        for j in tqdm(range(num_districts)):
            pops[j] = df[df.geoid.astype(str).apply(lambda i: bool(districts[j][i]))][run_cols].sum(axis=0).values
        print() 
        vs = np.var(pops, axis=1)
        vrs[split] = vs
    return vrs


def analytical_district_variances(h, districts, num_attributes=1):
    vrs = {}
    num_districts = len(districts)
    for split in ["equal", "top_heavy", "mid_heavy", "bottom_heavy", "bg_heavy"]:
        print(split)
        vs = np.array([h.assign_district_tree_variance(dist, eps=1, 
                         eps_splits=epsilon_split[split]) for dist in districts])*num_attributes
        vrs[split] = vs
    return vrs


def district_data_frame(frag_scores, model_variances, district_type):
    """
        frag_scores: array of fragmentation scores
        model_variances: list of (name, variance_dict) tuples
        district_type: str of district generation algorithm
    """
    results = pd.DataFrame()
    for split in ["equal", "top_heavy", "mid_heavy", "bottom_heavy", "bg_heavy"]:
        for model_name, vrs in model_variances:
            try:
                results = results.append(pd.DataFrame(vrs[split], 
                                                  columns=["variance"]).assign(frag_score=frag_scores, 
                                                                               split=split, 
                                                                               model=model_name, 
                                                                               eps=1,
                                                                               district_type=district_type))
            except:
                print("No {} weighting for {}".format(split, model_name))
    return results


tract_recom = np.load("sample_districts/100_recom_tract_parts_comp.npy", allow_pickle=True)[0]["dicts"]
block_recom = np.load("sample_districts/100_recom_block_parts_comp.npy", allow_pickle=True)[0]["dicts"]
block_bb = np.load("sample_districts/400_squre_block_dists.npy", allow_pickle=True)[0]["dicts"]
tract_discon = np.load("sample_districts/400_discon_tracts.npy", allow_pickle=True)
block_discon = np.load("sample_districts/400_discon_blocks.npy", allow_pickle=True)

# toydown_allow_neg = toydown_empirical_variances(tract_recom, neg_flag="allow_neg")
# np.save("variances/tract_recom_toydown_allow_neg.npy", [toydown_allow_neg], allow_pickle=True)


for district_type, ds in [("tract_recom", tract_recom), 
                          ("block_recom", block_recom),
                          ("block_bb", block_bb), ("tract_discon",tract_discon), 
                          ("block_discon", block_discon)]:
    print("Starting {} calculations".format(district_type), flush=True)
    frag_scores = district_frag_scores(h, ds)
    analytical_toydown = analytical_district_variances(h, ds, num_attributes=7)
    toydown_allow_neg = toydown_empirical_variances(ds, neg_flag="allow_neg")
    toydown_non_neg = toydown_empirical_variances(ds, neg_flag="non_neg")
    topdown_no_households = topdown_empirical_variances(ds, household_cons=False)
    topdown_with_households = topdown_empirical_variances(ds, household_cons=True)

    model_variances = [("ToyDown_analytical", analytical_toydown), 
                       ("ToyDown_allow_neg", toydown_allow_neg), 
                       ("ToyDown_non_neg",toydown_non_neg), 
                       ("TopDown_no_HH_cons", topdown_no_households), 
                       ("TopDown_with_HH_cons", topdown_with_households)]
    df = district_data_frame(frag_scores, model_variances, district_type)
    df.to_csv("variances/{}_fragscore_v_vars.csv".format(district_type), index=False)

