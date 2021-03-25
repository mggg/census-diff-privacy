import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from functools import partial
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
from matplotlib.backends.backend_pdf import PdfPages


SPLITS = ["equal", "mid_heavy", "bg_heavy", "bottom_heavy"]
models = ["ToyDown_allow_neg", "ToyDown_non_neg"]


"""
    Block Recom on Galveston City
"""

galv_city_errors = pd.DataFrame()

for split in SPLITS:
    df = pd.read_csv("dist_errors/galveston_city_block_recom_{}_split_errors.csv".format(split))
    galv_city_errors = galv_city_errors.append(df)

mean_galv_city_error = galv_city_errors.groupby(["dist_id", "model", "split", "district_type"]).mean().reset_index()


fig, axs = plt.subplots(4,len(models), figsize=(15,20))
xmin, xmax = mean_galv_city_error.ErrorMag.min(), mean_galv_city_error.ErrorMag.max(), 
for i, split in enumerate(SPLITS):
    for j, model in enumerate(models):
        df1 = mean_galv_city_error.query("split == @split and model==@model")
        sns.histplot(data=df1,
                     legend=False, bins=25, hue="district_type", 
                     x="ErrorMag", log_scale=True, element="step", ax=axs[i,j]
                     )
        axs[i,j].set_xlabel("mean error magnitude")
        axs[i,j].set_xlim(xmin, xmax)
        axs[i,j].get_yaxis().set_ticks([])
        axs[i,j].set_ylabel("")

pad = 5
for ax, row in zip(axs[:,0], ["{} split".format(s) for s in ["Equal", "Tract heavy", "Blockgroup heavy", "Block heavy"]]):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

for ax, col in zip(axs[0], ["Model: {}".format(m) for m in models]):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, ax.xaxis.labelpad + pad*4),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
fig.suptitle("6 Galveston City Council Districts (Block ReCom)")
fig.savefig("plots/galveston_city_districts_block_recom_mean_errormag_with_log_axis.png")
fig.show()


"""
    Tract Recom/Discon and on Bell, Brazoria, Cameron, Galveston, and Nueces Counties
"""
counties = ["bell","brazoria","cameron","galveston","nueces"]

for county in counties:
    county_prefix = "{}_county_".format(county)
    cnty_prefix_len = len(county_prefix)
    
    errors = pd.DataFrame()
    for split in SPLITS:
        for alg_type in ["recom", "discon"]:
            df = pd.read_csv("dist_errors/{}_county_tract_{}_{}_split_errors.csv".format(county, alg_type, split))
            errors = errors.append(df)
    mean_errors = errors.groupby(["dist_id", "model", "split", "district_type"]).mean().reset_index()
    mean_errors.district_type = mean_errors.district_type.apply(lambda s: s[cnty_prefix_len:])


    fig, axs = plt.subplots(4,len(models), figsize=(15,20))
    xmin, xmax = mean_errors.ErrorMag.min(), mean_errors.ErrorMag.max(), 

    for i, split in enumerate(SPLITS):
        for j, model in enumerate(models):
            df1 = mean_errors.query("split == @split and model==@model")
            sns.histplot(data=df1,
                        legend=False, bins=25, hue="district_type", 
                        x="ErrorMag", log_scale=True, element="step", ax=axs[i,j],
                        hue_order=["tract_recom", "tract_discon"])
            axs[i,j].set_xlabel("mean error magnitude")
            axs[i,j].set_xlim(xmin, xmax)
            axs[i,j].get_yaxis().set_ticks([])
            axs[i,j].set_ylabel("")

    pad = 5
    for ax, row in zip(axs[:,0], ["{} split".format(s) for s in ["Equal", "Tract heavy", "Blockgroup heavy", "Block heavy"]]):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    for ax, col in zip(axs[0], ["Model: {}".format(m) for m in models]):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, ax.xaxis.labelpad + pad*4),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    fig.suptitle("4 County Council Districts ({})".format(county.capitalize()))
    fig.savefig("plots/{}_county_districts_mean_errormag_with_log_axis.png".format(county))
    fig.show()