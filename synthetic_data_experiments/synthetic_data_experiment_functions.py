""" Synthetic Data Experiment functions
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from scipy import stats

def read_df_1940(person_file):
    """ Reads the data from the `person_file`. The columns were gathered from the Person line in
        the Census 1940s MDF writer here:
        https://github.com/uscensusbureau/census2020-das-e2e/blob/master/programs/writer/e2e_1940_writer.py
    """
    columns = ['SCHEMA_TYPE_CODE', 'SCHEMA_BUILD_ID', 'TABBLKST', 'TABBLKCOU', 'ENUMDIST',
               'EUID', 'EPNUM', 'RTYPE', 'QREL', 'QSEX', 'QAGE', 'CENHISP', 'CENRACE',
               'QSPANX', 'QRACE1', 'QRACE2', 'QRACE3', 'QRACE4', 'QRACE5', 'QRACE6', 'QRACE7',
               'QRACE8', 'CIT']
    df = pd.read_table(person_file, sep="|", header=None)
    df.columns = columns
    return df

def race_percents_by_district(df, state_id, race):
    """ Returns the % of people in the df of race `race` by district as a Dataframe
    """
    state = df[df["TABBLKST"] == state_id]
    pop_percent_df = pd.DataFrame(columns=["State", "County", "Enumdist"])

    for county in state["TABBLKCOU"].unique():
        enum_dists = state[state["TABBLKCOU"] == county]["ENUMDIST"].unique()
        for enum_dist in enum_dists:
            dist_df = state[(state["TABBLKCOU"] == county) & (state["ENUMDIST"] == enum_dist)]

            tot_pop = len(dist_df.index)
            race_pop = len(dist_df[dist_df["CENRACE"] == race].index)

            pop_percent_df = pop_percent_df.append({"State":state_id,
                                                    "County":county,
                                                    "Enumdist":enum_dist,
                                                    "Run":float(race_pop)/tot_pop},
                                                    ignore_index=True)
    return pop_percent_df

def rename_county_and_enumdist_to_input_names(df):
    """ Renamed the outputs to match the names of the inputs. This convolution is because I picked up the
        Person lines for the experiments straight from the 1940s ipums file and thus had to map those county
        ids to the ones we have in our synthetic data.
    """
    df.loc[df["County"] == 10, "County"] = 11
    df.loc[df["County"] == 30, "County"] = 12
    df.loc[df["County"] == 50, "County"] = 13

    df.loc[(df["County"] == 11) & (df["Enumdist"] == 11), "Enumdist"] = 111
    df.loc[(df["County"] == 11) & (df["Enumdist"] == 12), "Enumdist"] = 112
    df.loc[(df["County"] == 11) & (df["Enumdist"] == 20), "Enumdist"] = 113

    df.loc[(df["County"] == 12) & (df["Enumdist"] == 10), "Enumdist"] = 121
    df.loc[(df["County"] == 12) & (df["Enumdist"] == 20), "Enumdist"] = 122
    df.loc[(df["County"] == 12) & (df["Enumdist"] == 30), "Enumdist"] = 123

    df.loc[(df["County"] == 13) & (df["Enumdist"] == 10), "Enumdist"] = 131
    df.loc[(df["County"] == 13) & (df["Enumdist"] == 20), "Enumdist"] = 132
    df.loc[(df["County"] == 13) & (df["Enumdist"] == 30), "Enumdist"] = 133

    df = df.sort_values(["County", "Enumdist"])
    df.reset_index(drop=True)
    return df

def pop_percents_by_race(person_file, state_id, race):
    """ Read and return the % of people of race `race` in person_file
    """
    df = read_df_1940(person_file)
    percents_by_race_df = race_percents_by_district(df, state_id, race)
    renamed_df = rename_county_and_enumdist_to_input_names(percents_by_race_df)
    return renamed_df

def collect_run_percents_by_race(dir_name, state_id, race):
    """ Generates a dataframe of the % of people of race `race` by district
        for all the runs in an experiment.
    """
    run = 0
    main_df = pd.DataFrame(columns=["State", "County", "Enumdist"])
    for root, dirs, files in os.walk(dir_name):
        for d in dirs:
            if d[:7] == "output_":
                path = os.path.join(root, d)
                person_file = path + "/MDF_PER_CLEAN.dat"
                run += 1

                percents_by_race_df = pop_percents_by_race(person_file, state_id, race)
                percents_by_race_df = percents_by_race_df.rename(columns={"Run": "Run_{}".format(run)})

                main_df = pd.merge(main_df, percents_by_race_df, how="outer", on=["State", "County", "Enumdist"])
    return main_df

def plot_simulated_data(runs_df, num_runs, axs, plt_coords):
    """ Plot all the points using the runs in `runs_df` as x coordinates.
        Args:
            runs_df (Pandas DataFrame): df containing the runs, each run is a
                                        seperate column labeled for eg. "Run_1"
            num_runs: (int) Number of Runs in `runs_df`
            axs: (MatplotLib Axes object)
            plt_coords: (tuple) postion the graph is to be positioned in `axes`
    """
    ms = np.array([])
    rgba = (0.9375,0.5,0.5,.5)
    _, percent_votes_for_X = get_original_data()

    for i in range(1, num_runs+1):
        # plot the points in the run
        percents_by_race = np.array(runs_df["Run_{}".format(i)])
        axs[plt_coords[0], plt_coords[1]].scatter(percents_by_race,
                                                  percent_votes_for_X,
                                                  c=[rgba,])

        slope, intercept, _, _, _ = stats.linregress(percents_by_race, percent_votes_for_X)
        ms = np.append(ms, slope)

        if i == num_runs:
            # hacky: label the last line
            mean = "{0:.3g}".format(np.mean(ms))
            var = "{0:.3g}".format(np.var(ms))

            axs[plt_coords[0], plt_coords[1]].plot(percents_by_race,
                                                   intercept + slope*percents_by_race,
                                                   c=rgba,
                                                   label="E(m): {mean}, Var(m): {var}".format(mean=mean,
                                                                                              var=var))
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
    percent_votes_for_X = np.array([0.15645235, 0.20648465, 0.23917829, 0.44568536,
                                    0.51541049, 0.68753399, 0.73400675, 0.82867988,
                                    0.89909448])

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
    mean = "{0:.3g}".format(slope)

    axs[plt_coords[0], plt_coords[1]].plot(race_percents,
                                           intercept + race_percents*slope,
                                           c=(0,0,1,1),
                                           label="m: {mean}".format(mean=mean)
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
    slope, _, _, _, _ = stats.linregress(race_percents, percent_votes_for_X)
    axs[plt_coords[0], plt_coords[1]].axvline(x=slope, color="b", zorder=10)

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
        slope, _, _, _, _ = stats.linregress(percents_by_race, percent_votes_for_X)
        ms = np.append(ms, slope)

    axs[plt_coords[0], plt_coords[1]].hist(ms, color="lightcoral", bins=np.linspace(-0.5, 1.2, 35))

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

    axs[plt_coords[0], plt_coords[1]].set_xlabel("Slope of ER line")
    axs[plt_coords[0], plt_coords[1]].set_ylabel("Frequency")
    axs[plt_coords[0], plt_coords[1]].set_xlim(0.2, 1.6)
    axs[plt_coords[0], plt_coords[1]].set_ylim(0, 50)

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
