from gerrychain.random import random
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
from tqdm import tqdm
# from local_tools.randomDistricts import Hierarchy_2D

def sample_recom_partitions(init_part, ideal_pop, pop_col, eps, tot_steps=5000,
                            compactness=False, subsamp=None):
    proposal = partial(recom,
                    pop_col=pop_col,
                    pop_target=ideal_pop,
                    epsilon=eps,
                    node_repeats=1)
    pop_constraint = constraints.within_percent_of_ideal_population(init_part, eps)
    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]),
        2*len(init_part["cut_edges"])
    )

    chain = MarkovChain(
        proposal=proposal,
        constraints=[pop_constraint, compactness_bound] if compactness else [pop_constraint],
        accept=accept.always_accept,
        initial_state=init_part,
        total_steps=tot_steps
    )
    
    num_samps = int(np.ceil(tot_steps / subsamp if subsamp else tot_steps))
    # frag_scores = np.zeros(num_samps)
    districts = np.zeros(num_samps, dtype="object")
    parts = np.zeros(num_samps, dtype="object")
    j = 0
    for i, p in enumerate(chain.with_progress_bar()):
        if subsamp and (i % subsamp == subsamp-1):
            districts[j] = p["district_choice"]
            # frag_scores[j] = p["frag_score"]
            parts[j] = p
            j += 1
        elif not subsamp:
            districts[i] = p["district_choice"]
            # frag_scores[i] = p["frag_score"]
    return districts, parts

def dist_choices(df):
    def tract_dist(partition):
        dist_choice = np.random.randint(num_dists)
        graph = partition.graph
        tracts = [graph.nodes()[n]["GEOID10"] for n in partition.parts[dist_choice]]
        dist = {b: b[:-4] in tracts for b in df.GEOID10}
        return dist

    def bg_dist(partition):
        dist_choice = np.random.randint(num_dists)
        graph = partition.graph
        bgs = [graph.nodes()[n]["BG"] for n in partition.parts[dist_choice]]
        dist = {b: b[:-3] in bgs for b in df.GEOID10}
        return dist

    def block_dist(partition):
        dist_choice = np.random.randint(num_dists)
        graph = partition.graph
        blocks = [graph.nodes()[n]["GEOID10"] for n in partition.parts[dist_choice]]
        dist = {b: b in blocks for b in df.GEOID10}
        return dist

    return {"tract": tract_dist, "bg": bg_dist, "block": block_dist}


def random_discontiguous_district(gdf, ideal_pop, pop_min, pop_col, block_df, tracts=False, bgs=False):
        make_dist = True
        
        while(make_dist):
            node_order = gdf.sample(frac=1)
            d_pop = node_order[pop_col].cumsum()
            dist_df = node_order[d_pop < ideal_pop]
            
            make_dist = dist_df[pop_col].sum() < pop_min
        
        if tracts:
            tract_ids = set(dist_df.reset_index().GEOID10)
            district = {g: 1 if g[:-4] in tract_ids else 0 for g in block_df.GEOID10}
        elif bgs:
            bgs_ids = set(dist_df.reset_index().BG)
            district = {g: 1 if g[:-3] in bgs_ids else 0 for g in block_df.GEOID10}
        else:
            district = {g: 0 for g in block_df.GEOID10}
            for g in dist_df.GEOID10:
                district[g] = 1

        return district

"""
    Block Recom on Galveston City
"""
num_dists = 6
galveston_city_blocks = gpd.read_file("shapes/blocks/galveston_city_blocks_2010_data.shp")
galveston_city_graph = Graph.from_file("shapes/blocks/galveston_city_blocks_2010_data.shp")

pop_col = "TOTPOP"
total_pop = galveston_city_blocks[pop_col].sum()
ideal_pop = total_pop / num_dists
eps = 0.02

cdict = tree.recursive_tree_part(galveston_city_graph, range(num_dists), ideal_pop, pop_col, eps, node_repeats=1)
init_part = Partition(galveston_city_graph, cdict, 
                      updaters={"cut_edges": cut_edges, "population": Tally(pop_col, alias="population"),
                                "district_choice": dist_choices(galveston_city_blocks)["block"]})

block_dists, parts = sample_recom_partitions(init_part, ideal_pop,  pop_col, eps, tot_steps=10000, 
                                             compactness=False, subsamp=10)

np.save("sample_districts/galveston_city_1000_recom_block_dists_{}_eps_{}.npy".format(num_dists, eps), block_dists)

"""
    Tract Recom/Discon and on Bell, Brazoria, Cameron, Galveston, and Nueces Counties
"""
num_dists = 4
counties = ["bell","brazoria","cameron","galveston","nueces"]

for county in counties:
    blocks = gpd.read_file("shapes/blocks/{}_county_blocks_2010_data.shp".format(county))
    tracts = gpd.read_file("shapes/tracts/{}_county_tract_2010_data.shp".format(county))
    tract_g = Graph.from_file("shapes/tracts/{}_county_tract_2010_data.shp".format(county))
    pop_col = "TOTPOP"
    total_pop = tracts[pop_col].sum()
    ideal_pop = total_pop / num_dists
    eps = 0.02
    
    print("making ReCom ensemble")
    cdict = tree.recursive_tree_part(tract_g, range(num_dists), ideal_pop, pop_col, eps, node_repeats=1)
    init_part = Partition(tract_g, cdict, 
                          updaters={"cut_edges": cut_edges, "population": Tally(pop_col, alias="population"),
                                    "district_choice": dist_choices(blocks)["tract"]})
    tract_dists, tract_parts = sample_recom_partitions(init_part, ideal_pop,  pop_col, eps, tot_steps=10000, 
                                                 compactness=False, subsamp=10)
    np.save("sample_districts/{}_county_1000_recom_tract_dists_{}_eps_{}.npy".format(county, num_dists, eps), tract_dists)

    print("making Discon ensemble")
    pop_min, pop_max = (ideal_pop*(1-eps), ideal_pop*(1+eps))
    districts = np.zeros(1000, dtype="object")
    for i in range(1000):
        if i % 20 == 0:
            print("*", flush=True, end="")
        dist = random_discontiguous_district(tracts, ideal_pop, pop_min, pop_col, blocks,
                                            tracts=True)
        districts[i] = dist
    print()
    np.save("sample_districts/{}_county_1000_discon_tract_dists_{}_eps_{}.npy".format(county, num_dists, eps), districts)
