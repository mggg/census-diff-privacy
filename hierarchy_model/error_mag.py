import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from functools import partial
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
from matplotlib.backends.backend_pdf import PdfPages


tract_recom1000_vars = pd.read_csv("variances/tract_fragscore_v_vars_1000.csv")
block_recom1000_vars = pd.read_csv("variances/block_fragscore_v_vars_1000.csv")
bg_recom1000_vars = pd.read_csv("variances/bg_fragscore_v_vars_1000.csv")
pct_recom1000_vars = pd.read_csv("variances/pct_fragscore_v_vars_1000.csv")
block_bb_vars = pd.read_csv("variances/block_bb_fragscore_v_vars.csv")
tract_discon_vars = pd.read_csv("variances/tract_discon_fragscore_v_vars.csv")
block_discon_vars = pd.read_csv("variances/block_discon_fragscore_v_vars.csv")


df_vars = tract_recom1000_vars.append(block_recom1000_vars).append(tract_discon_vars).append(block_discon_vars)
df_vars = df_vars.replace({"district_type": {"tract": "tract_recom", "block": "block_recom"}})
df_vars["weight"] = df_vars["district_type"].apply(lambda t: 4/10 if t.endswith("recom") else 1)
df_vars["standard_deviation"] = df_vars["variance"].apply(np.sqrt)

epsilon_splits = ["equal", "mid_heavy", "bg_heavy", "bottom_heavy"]
models = ["ToyDown_allow_neg", "ToyDown_non_neg", "TopDown_no_HH_cons"]
fig, axs = plt.subplots(4,3, figsize=(15,20))
for i, split in enumerate(epsilon_splits):
    for j, model in enumerate(models):
        sns.histplot(data=df_vars.query("split == @split and model==@model"),
                     legend=(i==0 and j==2), bins=25, hue="district_type", 
                     x="variance", log_scale=True, element="step", ax=axs[i,j], weights="weight")
        axs[i,j].set_xlabel("error variance")
        axs[i,j].set_xlim(10**4 / 3, 10**8 *2)
        axs[i,j].get_yaxis().set_ticks([])
        axs[i,j].set_ylabel("")

pad = 5
for ax, row in zip(axs[:,0], ["Split = {}".format(s) for s in epsilon_splits]):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

for ax, col in zip(axs[0], ["Model: {}".format(m) for m in models]):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, ax.xaxis.labelpad + pad*4),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
# fig.savefig("vars_with_log_axis.png")
fig.show()


errors = pd.DataFrame()
for alg in  ["block_recom_175", "block_discon_175", "bg_recom_175", "bg_discon_175"]: #["tract_discon", "tract_recom_4", "block_recom", "block_discon"]:
    if alg in ["tract_recom_4", "block_recom_175", "block_discon_175", "bg_recom_175", "bg_discon_175"]:
        errors = errors.append(pd.read_csv("dist_errors/{}_equal_split_errors.csv".format(alg)).query("dist_id < 'tract_recom_4_400'"))
        errors = errors.append(pd.read_csv("dist_errors/{}_bottom_heavy_split_errors.csv".format(alg)).query("dist_id < 'tract_recom_4_400'"))
        errors = errors.append(pd.read_csv("dist_errors/{}_mid_heavy_split_errors.csv".format(alg)).query("dist_id < 'tract_recom_4_400'"))
        errors = errors.append(pd.read_csv("dist_errors/{}_bg_heavy_split_errors.csv".format(alg)).query("dist_id < 'tract_recom_4_400'"))
    else:
        errors = errors.append(pd.read_csv("{}_equal_split_errors.csv".format(alg)))
        errors = errors.append(pd.read_csv("{}_bottom_heavy_split_errors.csv".format(alg)))
        errors = errors.append(pd.read_csv("{}_mid_heavy_split_errors.csv".format(alg)))
        errors = errors.append(pd.read_csv("{}_bg_heavy_split_errors.csv".format(alg)))

mean_error = errors.groupby(["dist_id", "model", "split", "district_type"]).mean().reset_index()


eq_errors = pd.DataFrame()
for alg in ["tract_recom", "tract_discon",
            #"blockgroup_recom", "precinct_recom", "block_recom",
           # "block_discon", "tract_discon", "block_bounding_box"]:
            "block_recom", "block_discon"]:
    eq_errors = eq_errors.append(pd.read_csv("dist_errors/{}_10_equal_split_errors.csv".format(alg)))
    eq_errors = eq_errors.append(pd.read_csv("dist_errors/{}_20_equal_split_errors.csv".format(alg)))

    if "block" in alg:
        eq_errors = eq_errors.append(pd.read_csv("dist_errors/{}_175_equal_split_errors.csv".format(alg)))
    else:
        eq_errors = eq_errors.append(pd.read_csv("dist_errors/{}_4_equal_split_errors.csv".format(alg)))

mean_erroreq = eq_errors.groupby(["dist_id", "model", "split", "district_type"]).mean().reset_index()

epsilon_splits = ["equal", "mid_heavy", "bg_heavy" ,"bottom_heavy"]
models = ["ToyDown_allow_neg", "ToyDown_non_neg", "TopDown_no_HH_cons"]

fig, axs = plt.subplots(4,2, figsize=(15,20))
for i, split in enumerate(epsilon_splits):
    for j, model in enumerate(models):
        df1 = errors.query("split == @split and model==@model").replace({"ErrorMag": {0: 0.5}})
        sns.histplot(data=df1,
                     legend=(i==0 and j==1), bins=25, hue="district_type", 
                     x="ErrorMag", log_scale=True, element="step", ax=axs[i,j],
                     hue_order=["block_recom", "tract_discon", "block_discon"]
                     )
        axs[i,j].set_xlabel("error magnitude")
        axs[i,j].set_xlim(10**1, 10**5)
        axs[i,j].get_yaxis().set_ticks([])
        axs[i,j].set_ylabel("")

pad = 5
for ax, row in zip(axs[:,0], ["Split = {}".format(s) for s in epsilon_splits]):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

for ax, col in zip(axs[0], ["Model: {}".format(m) for m in models]):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, ax.xaxis.labelpad + pad*4),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
fig.savefig("errormag_with_log_axis.png")
fig.show()

fig, axs = plt.subplots(4,3, figsize=(15,20))
for i, split in enumerate(epsilon_splits):
    for j, model in enumerate(models):
        df1 = mean_error.query("split == @split and model==@model")
        sns.histplot(data=df1,
                     legend=False, bins=25, hue="district_type", 
                     x="ErrorMag", log_scale=True, element="step", ax=axs[i,j],
                    hue_order=["block_recom_175", "block_discon_175", "bg_recom_175", "bg_discon_175"]
                     )
        axs[i,j].set_xlabel("mean error magnitude")
        axs[i,j].set_xlim(10**1, 10**4/4)
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
fig.savefig("mean_errormag_with_log_axis_175_toy_no_legend.png")
fig.show()


fig, axs = plt.subplots(12,1, figsize=(7, 25))

algs = ["tract_recom_4", "tract_discon_4", "tract_recom_10", "tract_discon_10","tract_recom_20", 
        "tract_discon_20",
        "block_recom_10", "block_discon_10","block_recom_20", "block_discon_20","block_recom_175", 
        "block_discon_175",]# "blockgroup_recom", "precinct_recom","block_recom",]


for i, alg in enumerate(algs):
    df_d = mean_erroreq.query("split == 'equal' and district_type == @alg")
    # df_d = eq_errors.query("split == 'equal' and district_type == @alg")
    g = sns.histplot(data=df_d, x="ErrorMag", element="step", ax=axs[i], hue="model",
                hue_order=["ToyDown_allow_neg","ToyDown_non_neg", "TopDown_no_HH_cons"], 
                legend=i==0, binwidth=75)
    # sns.histplot(data=df_d.query("model == 'ToyDown_allow_neg'"), x="frag_score", element="step", 
    #             ax=axs[i,0], binwidth=25)
    axs[i].set_xlim(-100, 2500)
    # axs[i,0].set_xlim(70, 1000)
    # axs[i,1].set_ylim(0, 240)
    axs[i].set_ylim(0, 700)
    # [(axs[i,j].get_yaxis().set_ticks([]), axs[i,j].set_ylabel("")) for j in [0,1]]
    # if i == 0:
        # axs[i].legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)


    


pad = 5
for ax, row in zip(axs, ["{}".format(s) for s in algs]):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
# for ax, col in zip(axs[0], ["{}".format(m) for m in ["Frag Score", "Mean Error Magnitude"]]):
#     ax.annotate(col, xy=(0.5, 1), xytext=(0, ax.xaxis.labelpad + pad*4),
#                 xycoords='axes fraction', textcoords='offset points',
#                 size='large', ha='center', va='baseline')
axs[0].annotate("Mean Error Magnitude", xy=(0.5, 1), xytext=(0, ax.xaxis.labelpad + pad*4),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')


fig.savefig("mean_error_mag_and_frag_score_equal_split_debug_with_small dist.png", bbox_inches='tight', dpi=200)



algs = ["tract_recom_4",  "tract_recom_10", "tract_recom_20", 
        "tract_discon_4","tract_discon_10", "tract_discon_20",
        "block_recom_10", "block_recom_20", "block_recom_175", 
        "block_discon_10","block_discon_20","block_discon_175"]

df_d = mean_erroreq.query("split == 'equal'")
g = sns.FacetGrid(df_d, col="district_type",hue="model", col_order=algs, col_wrap=3, xlim=(0,2500),
                hue_order=["ToyDown_allow_neg","ToyDown_non_neg", "TopDown_no_HH_cons"], legend_out=True)
g.map(sns.histplot, "ErrorMag", binwidth=75, alpha=0.5)
g.set_axis_labels("Mean Error Magnitude (TOTPOP)", "Count")
g.set_titles(col_template="{col_name}")
g.add_legend()
g.savefig("mean_error_mag_and_frag_score_equal_split_debug_with_small_dist.png", bbox_inches='tight', dpi=200)


np.load("sample_districts/1000_recom_tract_frags.npy").min()
np.load("sample_districts/1000_recom_tract_frags.npy").max()

np.load("sample_districts/1000_recom_block_frags.npy")

pd.read_csv("tract_discon_equal_split_errors.csv").frag_score.min()
df = pd.read_csv("tract_discon_equal_split_errors.csv").groupby(["dist_id", "model", "split", "district_type"]).mean().reset_index()

df.query("model == 'ToyDown_allow_neg'").ErrorMag.min()
df.query("model == 'ToyDown_allow_neg'").ErrorMag.max()

df.query("model == 'TopDown_no_HH_cons'").ErrorMag.min()
df.query("model == 'TopDown_no_HH_cons'").ErrorMag.max()

plt.hist(df.query("model == 'ToyDown_allow_neg'").ErrorMag)

plt.hist(df.query("model == 'TopDown_no_HH_cons'").ErrorMag)