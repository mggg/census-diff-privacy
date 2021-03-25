"""
    Synthetic Data Experiment functions
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from scipy import stats

def sort_df_by_county_dist(df, keys):
    """ Sorts `df` based on the order of `keys`.
        `keys` is expected to be an array of tuples of form (county, dist).

        `df` is sorted such that the first row is the first (county, dist) in
        the list, the second row is the second (county, dist), and so on.

        Assumes that `df` contains the columns "County" and "Enumdist", and also that
        all the (countuy, dist) tuples exist in the `df`.
    """
    idx = 0
    for (county, dist) in keys:
        idx += 1
        df.loc[(df["County"] == county) & (df["Enumdist"] == dist), "rank"] = idx
    df = df.sort_values(by=["rank"])
    df = df.drop(columns=["rank"])
    return df

def plot_simulated_data(runs_df, num_runs, axs, plt_coords):
    """ Plot all the points using the runs in `runs_df` as x coordinates.
        Args:
            runs_df (Pandas DataFrame): df containing the runs, each run is a
                                        seperate column labeled for eg. "Run_1"
            num_runs: (int) Number of Runs in `runs_df`
            axs: (MatplotLib Axes object)
            plt_coords: (tuple) postion the graph is to be positioned in `axes`
    """
    supports = np.array([])
    rgba = (0.9375,0.5,0.5,.5)
    _, percent_votes_for_X = get_original_data()

    for i in range(1, num_runs+1):
        # plot the points in the run
        percents_by_race = np.array(runs_df["Run_{}".format(i)])
        axs[plt_coords[0], plt_coords[1]].scatter(percents_by_race,
                                                  percent_votes_for_X,
                                                  c=[rgba,])

        slope, intercept, _, _, _ = stats.linregress(percents_by_race,
                                                     percent_votes_for_X)
        # support = mx + x at x = 1
        supports = np.append(supports, slope * 1 + intercept)

        if i == num_runs:
            # hacky: label the last line
            mean = "{0:.3g}".format(np.mean(supports))
            var = "{0:.3g}".format(np.var(supports))
            label = "E(support): {mean}, Var(support): {var}".format(mean=mean,
                                                                     var=var)

            axs[plt_coords[0], plt_coords[1]].plot(percents_by_race,
                                                   intercept + slope*percents_by_race,
                                                   c=rgba,
                                                   label=label)
        else:
            axs[plt_coords[0], plt_coords[1]].plot(percents_by_race,
                                                   intercept + slope*percents_by_race,
                                                   c=rgba)

def get_original_data():
    """ This is the original data fabricated by JN which was noised by TopDown
        in the runs.

        Returns:
            race_percents (numpy arr): Percent of people of race A in each district
            percent_votes_for_X: (numpy arr): Percent of votes obtained by
                                              candidate X in each district
    """
    race_a = [60, 101, 102, 112, 100, 116, 120, 161, 138]
    tot_pops = [345, 366, 260, 289, 294, 279, 200, 211, 151]
    race_percents = np.array([float(race_a)/tot_pops for (race_a,tot_pops) in zip(race_a, tot_pops)])

    # This is the synthetic vote percent by district for candidate X that we generated.
    percent_votes_for_X = np.array([0.15645235, 0.20648465, 0.23917829,
                                    0.44568536, 0.51541049, 0.68753399,
                                    0.73400675, 0.82867988, 0.89909448])

    return race_percents, percent_votes_for_X

def plot_original_data(axs, plt_coords):
    """ Plot the ER line for the unnoised, fabricated 9 point dataset.

        Args:
            axs: (MatplotLib Axes object)
            plt_coords: (tuple) postion the graph is to be positioned in `axes`
    """
    race_percents, percent_votes_for_X = get_original_data()

    # plot the points
    axs[plt_coords[0], plt_coords[1]].scatter(race_percents,
                                              percent_votes_for_X,
                                              c=[(0,0,1,1),],
                                              zorder=10)

    # plot the best fit line
    slope, intercept, _, _, _ = stats.linregress(race_percents,
                                                 percent_votes_for_X)

    race_percents = np.insert(race_percents, 0,0)
    race_percents = np.append(race_percents, 1)
    support = "{0:.3g}".format(slope * 1 + intercept)

    axs[plt_coords[0], plt_coords[1]].plot(race_percents,
                                           intercept + race_percents*slope,
                                           c=(0,0,1,1),
                                           label="support: {support}".format(support=support)
                                           ,zorder=10)

def plot_original_data_hist(axs, plt_coords):
    """ Plot a vertical line for the unnoised, fabricated 9 point dataset in the
        histogram at  coordinates `plt_coords` of `axs`.

        Args:
            axs: (MatplotLib Axes object)
            plt_coords: (tuple) postion the graph is to be positioned in `axs`
    """
    race_percents, percent_votes_for_X = get_original_data()
    race_percents = np.array(race_percents)

    slope, _, _, _, _ = stats.linregress(race_percents,
                                         percent_votes_for_X)

    axs[plt_coords[0], plt_coords[1]].axvline(x=slope,
                                              color="b",
                                              zorder=10)


def plot_simulated_data_hist(runs_df, num_runs, axs, plt_coords):
    """ Plot a histogram for the TopDown data at coordinates `plt_coords` of `axs`.

        Args:
            axs: (MatplotLib Axes object)
            plt_coords: (tuple) postion the graph is to be positioned in `axs`
    """
    ms = np.array([])
    _, percent_votes_for_X = get_original_data()

    # collect the slopes
    for i in range(1, num_runs+1):
        percents_by_race = np.array(runs_df["Run_{}".format(i)])
        slope, _, _, _, _ = stats.linregress(percents_by_race,
                                             percent_votes_for_X)
        ms = np.append(ms, slope)

    axs[plt_coords[0], plt_coords[1]].hist(ms,
                                           color="lightcoral",
                                           bins=np.linspace(-0.5, 1.2, 35))

def plot_histograms(axs, plt_coords, runs_df, num_runs):
    """ Generates a histogram of the ensemble of TopDown noised runs and labels
        the original data on it.

        Args:
            axs: (MatplotLib Axes object)
            plt_coords: (tuple) postion the graph is to be positioned in `axs`
            runs_df (Pandas DataFrame): df containing the runs, each run is a
                                        seperate column labeled for eg. "Run_1"
            num_runs: (int) Number of Runs in `runs_df`
    """
    plot_original_data_hist(axs, plt_coords)
    plot_simulated_data_hist(runs_df, num_runs, axs, plt_coords)

    axs[plt_coords[0], plt_coords[1]].set_xlabel("Support for Candidate X")
    axs[plt_coords[0], plt_coords[1]].set_ylabel("Frequency")
    axs[plt_coords[0], plt_coords[1]].set_xlim(0.2, 1.6)
    axs[plt_coords[0], plt_coords[1]].set_ylim(0, 70)

def plot_er(axs, plt_coords, runs_df, num_runs):
    """ Generates ER lines of the ensemble of TopDown noised runs and labels
        the original data on it.

        Args:
            axs: (MatplotLib Axes object)
            plt_coords: (tuple) postion the graph is to be positioned in `axs`
            runs_df (Pandas DataFrame): df containing the runs, each run is a
                                        seperate column labeled for eg. "Run_1"
            num_runs: (int) Number of Runs in `runs_df`
    """
    plot_simulated_data(runs_df, num_runs, axs, plt_coords)
    plot_original_data(axs, plt_coords)
    axs[plt_coords[0], plt_coords[1]].set_xlabel("Percent A")
    axs[plt_coords[0], plt_coords[1]].set_ylabel("Percent vote for X")
    axs[plt_coords[0], plt_coords[1]].hlines([0, 1], 0, 1, linestyles='dashed')
    axs[plt_coords[0], plt_coords[1]].legend()

def plot(runs_df, num_runs, axs, plt_coords, hist=False):
    """ General function that plots the data in `runs_df` in the
        `plt_coords` position of `axes`.
        Plots ecological regression plots by default.
    """
    if hist:
        plot_histograms(axs, plt_coords, runs_df, num_runs)
    else:
        plot_er(axs, plt_coords, runs_df, num_runs)
