import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import geopandas as gpd
from sklearn.linear_model import LinearRegression


## Compare ER with ToyDown/TopDown noised data
def plot_er_graph(data, cand, race, elect, elect_col, tot_vote, eps, split, filt=True, n_samps=32, 
                  ax=None, title=True, plot_cvap=False, weight=False):
    
    df = data.query("epsilon == @eps & split == @split")
    df = df.query("`{}` > 10".format(tot_vote)) if filt else df
    num_precints = df.shape[0]
    xs = np.reshape(np.linspace(0, 1, 100), (100,1))
    perc_race = np.reshape(df["{}_pct".format(race)].fillna(0).values, (num_precints, 1))
    perc_cand = df["{}{}".format(cand,elect_col)].fillna(0) 
    weights = df[tot_vote].values if weight else None
    line = LinearRegression().fit(perc_race, perc_cand, weights)

    ms = np.zeros(n_samps)

    if ax==None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1, 1, 1)
    if title: ax.set_title("ER - Votes for {}: {}".format(cand, elect))
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05)

    for i in range(n_samps):
        perc_race_noised = np.reshape((df["{}_{}_noise".format(i, race)] / df["{}_VAP_noise".format(i)]).fillna(0).values,
                                      (num_precints,1))
            
        line_noised = LinearRegression().fit(perc_race_noised, perc_cand, weights)
        ms[i] = line_noised.coef_[0]
        ax.plot(perc_race_noised, perc_cand, '.', color="lightcoral")
        ax.plot(xs, line_noised.predict(xs), '-', color="red")
    
    ax.plot(perc_race, perc_cand, '.', color="b")
    ax.plot(xs, line.predict(xs), '-', color="b", 
             label="m: {}".format(round(line.coef_[0], 3)))
    ax.plot([], [], '-', color="r", 
             label="E(m): {}, Var(m): {}".format(round(np.mean(ms),3), round(np.var(ms),4)))  
    
    ax.legend()
    ax.set_xlabel("% {}".format(race))
    ax.set_ylabel("{} % of Voters".format(cand))
    return ax

## Plot grid of epsilon values vs. splits for ToyDown/TopDown ER plots
def plot_elect_grid(epsilon_values, epsilon_splits, data, candidate, race, elect_col, 
                    tot_vote, figsize=(10,10), filt=True, title=None, weight=False, n_samps=32):
    
    fig, axs = plt.subplots(len(epsilon_values),len(epsilon_splits), figsize=figsize)

    if title: fig.suptitle(title)
    plt.subplots_adjust(hspace = 0.25)

    for i in range(len(epsilon_values)):
        for j in range(len(epsilon_splits)):
            plot_er_graph(data, candidate, race, None, elect_col, tot_vote, 
                          epsilon_values[i], epsilon_splits[j], 
                          title=False, ax=axs[i,j], filt=filt, weight=weight, n_samps=n_samps)

    pad = 5
    for ax, row in zip(axs[:,0], ["$\epsilon$ = {}".format(eps) for eps in epsilon_values]):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    for ax, col in zip(axs[0], ["Split: {}".format(s) for s in epsilon_splits]):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, ax.xaxis.labelpad + pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
        
    return fig, axs


## Get Point estimates of unnoised data.
def point_estimates(data, cand, race, elect_col, tot_vote, eps, split, filt=True, weight=False):
    
    df = data.query("epsilon == @eps & split == @split")
    df = df.query("`{}` > 10".format(tot_vote)) if filt else df
    num_precints = df.shape[0]
    xs = np.reshape(np.linspace(0, 1, 100), (100,1))
    perc_race = np.reshape(df["{}_pct".format(race)].fillna(0).values, (num_precints, 1))
    perc_cand = df["{}{}".format(cand, elect_col)].fillna(0) 
    weights = df[tot_vote].values if weight else None
    line = LinearRegression().fit(perc_race, perc_cand, weights)
    
    return line.predict([[0],[1]])


def plot_point_estimates(data, cand, race, elect, elect_col, tot_vote, eps, split,
                         filt=True, n_samps=32, ax=None, title=True, x_lims=None, weight=False):
    
    df = data.query("epsilon == @eps & split == @split")
    df = df.query("`{}` > 10".format(tot_vote)) if filt else df
    num_precints = df.shape[0]
    xs = np.reshape(np.linspace(0, 1, 100), (100,1))
    perc_race = np.reshape(df["{}_pct".format(race)].fillna(0).values, (num_precints, 1))
    perc_cand = df["{}{}".format(cand,elect_col)].fillna(0) 
    weights = df[tot_vote].values if weight else None
    line = LinearRegression().fit(perc_race, perc_cand, weights)

    ms = np.zeros(n_samps)
    zeros = np.zeros(n_samps)
    ones = np.zeros(n_samps)

    if ax==None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1, 1, 1)
    if title: ax.set_title("ER Point Estimates - Votes for {}: {}".format(cand, elect))

    if x_lims: ax.set_xlim(x_lims[0],x_lims[1])

    
    for i in range(n_samps):
        perc_race_noised = np.reshape((df["{}_{}_noise".format(i, race)] / df["{}_VAP_noise".format(i)]).fillna(0).values,
                                      (num_precints,1))
            
        line_noised = LinearRegression().fit(perc_race_noised, perc_cand, weights)
        ms[i] = line_noised.coef_[0]
        zeros[i] = line_noised.intercept_
        ones[i] = line_noised.predict([[1]])[0]
    
    ax.hist(zeros, color="limegreen", alpha=0.5, label="all but {} support".format(race))
    ax.hist(ones, color="mediumpurple",  alpha=0.5, label="{} support".format(race))
    ax.axvline(zeros.mean(), color="limegreen")
    ax.axvline(ones.mean(), color="mediumpurple")
    ax.axvline(line.intercept_, color="slategrey", linestyle="dashed")
    ax.axvline(line.predict([[1]])[0], color="slategrey", linestyle="dashed", label="un-noised data")
    
    ax.legend(loc="upper center")
    ax.set_xlabel("support for {}".format(cand))
    return ax

def plot_point_estimate_grid(epsilon_values, epsilon_splits, data, candidate, race, elect_col, 
                    tot_vote, figsize=(10,10), filt=True, title=True, x_lims=None, weight=False, n_samps=32):
    
    fig, axs = plt.subplots(len(epsilon_values),len(epsilon_splits), figsize=figsize)

    if title: fig.suptitle(title)
    plt.subplots_adjust(hspace = 0.25)

    for i in range(len(epsilon_values)):
        for j in range(len(epsilon_splits)):
            plot_point_estimates(data, candidate, race, None, elect_col, tot_vote, 
                                 epsilon_values[i], epsilon_splits[j], weight=weight,
                                 title=False, ax=axs[i,j], filt=filt, x_lims=x_lims, n_samps=n_samps)

    pad = 5
    for ax, row in zip(axs[:,0], ["$\epsilon$ = {}".format(eps) for eps in epsilon_values]):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    for ax, col in zip(axs[0], ["Split: {}".format(s) for s in epsilon_splits]):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, ax.xaxis.labelpad + pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
        
    return fig, axs


def plot_er_graph_gaussian_noise(data, cand, race, elect, tot_vote, sigma,
                  filt=True, n_samps=32, ax=None, title=True, plot_cvap=False, weight=False):
    
    df = data.query("`{}` > 10".format(tot_vote)) if filt else data
    num_precints = df.shape[0]
    xs = np.reshape(np.linspace(0, 1, 100), (100,1))
    perc_race = np.reshape((df[race] / df["VAP"]).fillna(0).values, (num_precints, 1))
    perc_cand = df["{}D_18R_Governor_pct".format(cand)].fillna(0)
    weights = df["18R_Governor_D_tot"].values if weight else None
    line = LinearRegression().fit(perc_race, perc_cand, weights)


    ms = np.zeros(n_samps)

    if ax==None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1, 1, 1)
    if title: ax.set_title("ER - Votes for {}: {}".format(cand, elect))
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05)

    for i in range(n_samps):
        noise = lambda sigma: np.random.normal(scale=sigma, size=df.shape[0])
        perc_race_noised = np.reshape(((df[race] + noise(sigma)) / (df["VAP"] + noise(sigma))).fillna(0).values, 
                                      (num_precints, 1))
            
        line_noised = LinearRegression().fit(perc_race_noised, perc_cand, weights)
        ms[i] = line_noised.coef_[0]
        ax.scatter(perc_race_noised, perc_cand, marker='.', color="lightcoral", s=weights, alpha=0.5)
        ax.plot(xs, line_noised.predict(xs), '-', color="red")
    
    ax.scatter(perc_race, perc_cand, marker='.', color="b", s=weights, alpha=0.5)
    ax.plot(xs, line.predict(xs), '-', color="navy", 
             label="m: {}".format(round(line.coef_[0], 3)))
    ax.plot([], [], '-', color="r", 
             label="E(m): {}, Var(m): {}".format(round(np.mean(ms),3), round(np.var(ms),4)))  
    
    ax.legend()
    ax.set_xlabel("% {}".format(race))
    ax.set_ylabel("{} % of Voters".format(cand))
    return ax

def plot_elect_grid_gaussian_noise(epsilon_values, epsilon_splits, data, candidate, race, election, 
                                   tot_vote, sigma_matches, allow_neg=False, figsize=(10,10), filt=True, 
                                   title=None, weight=False, n_samps=32):
    
    fig, axs = plt.subplots(len(epsilon_values),len(epsilon_splits), figsize=figsize)

    if title: fig.suptitle(title)
    plt.subplots_adjust(hspace = 0.25)

    for i in range(len(epsilon_values)):
        for j in range(len(epsilon_splits)):
            eps = epsilon_values[i]
            split = epsilon_splits[j]
            sigma = sigma_matches.query("allow_neg == @allow_neg and eps == @eps and split == @split").sigma
            plot_er_graph_gaussian_noise(data, candidate, race, election, tot_vote, 
                                         sigma, title=False, ax=axs[i,j], filt=filt, weight=weight, n_samps=n_samps)
            axs[i,j].set_title("$\sigma$ = {}".format(round(float(sigma), 2)))

    pad = 5
    for ax, row in zip(axs[:,0], ["$\epsilon$ = {}".format(eps) for eps in epsilon_values]):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    for ax, col in zip(axs[0], ["Split: {}".format(s) for s in epsilon_splits]):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, ax.xaxis.labelpad + pad*4),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
        
    return fig, axs