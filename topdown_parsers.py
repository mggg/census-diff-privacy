""" A library for functions to read and process TopDown 1940 outputs.
    The repo that generates these outputs is https://github.com/uscensusbureau/census2020-das-e2e
"""
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

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

def values_by_enumdist_for_run(df, state_id, race=None, race_percent=False):
    """ Generates and returns a DataFrame that tabulates "values" by enumdist
        for a single run.

        By default, this value is the total population of the enumdist.
        If `race` is passed, then the total population of that race by enumdist
        is computed.
        If `race_percent` is passed, the % of the race in each enumdist is
        returned.
    """
    state = df[df["TABBLKST"] == state_id]
    values_df = pd.DataFrame(columns=["State", "County", "Enumdist"])

    for county in state["TABBLKCOU"].unique():
        enum_dists = state[state["TABBLKCOU"] == county]["ENUMDIST"].unique()
        for enum_dist in enum_dists:
            dist_df = state[(state["TABBLKCOU"] == county) & (state["ENUMDIST"] == enum_dist)]
            tot_pop = len(dist_df.index)

            if race and race_percent:
                race_pop = len(dist_df[dist_df["CENRACE"] == race].index)
                value = float(race_pop)/tot_pop
            elif race:
                race_pop = len(dist_df[dist_df["CENRACE"] == race].index)
                value = race_pop
            else:
                value = tot_pop

            values_df = values_df.append({"State":state_id,
                                          "County":county,
                                          "Enumdist":enum_dist,
                                          "Run": value},
                                          ignore_index=True)
    return values_df

def collect_by_enumdist(dir_name,
                        state_id,
                        race=None,
                        race_percent=False):
    """ Generates and returns a DataFrame that tabulates "values" by enumdist for
        all runs.

        By default, this value is the total population of the enumdist.
        If `race` is passed, then the total population of that race by enumdist
        is computed.
        If `race_percent` is passed, the % of the race in each enumdist is
        returned.
    """
    if race_percent and not race:
        raise Exception("Keyword Arg `race_percent` cannot be True if " \
                        "`race` is not passed.")

    run = 0
    main_df = pd.DataFrame(columns=["State", "County", "Enumdist"])
    for root, dirs, files in os.walk(dir_name):
        for d in dirs:
            if d[:7] == "output_":
                path = os.path.join(root, d)
                person_file = path + "/MDF_PER_CLEAN.dat"

                person_df = read_df_1940(person_file)
                values_df = values_by_enumdist_for_run(person_df,
                                                       state_id,
                                                       race,
                                                       race_percent)

                run += 1
                values_df= values_df.rename(columns={"Run": "Run_{}".format(run)})

                # combine with the major
                main_df = pd.merge(main_df,
                                   values_df,
                                   how="outer",
                                   on=["State", "County", "Enumdist"])
    return main_df
