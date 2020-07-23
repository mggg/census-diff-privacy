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

    if race and race_percent:
        tot_pops = state.groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"]).size().reset_index()
        tot_pops.columns = ["TABBLKST", "TABBLKCOU", "ENUMDIST", "Run"]

        state['race_match'] = state.CENRACE == race
        values_df = state.groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"])['race_match'].agg([('Run', 'sum')]).reset_index()
        values_df["Run"]  = values_df["Run"] / tot_pops["Run"]
    elif race:
        state['race_match'] = state.CENRACE == race
        values_df = state.groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"])['race_match'].agg([('Run', 'sum')]).reset_index()
    else:
        values_df = state.groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"]).size().reset_index()

    values_df.columns = ["State", "County", "Enumdist", "Run"]
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

def pops_by_enumdist(input_file, state_fips_code):
    """ Returns the population in each enumdist of the state `state_fips_code`.
        `input_file` is the filepath of the .dat file used as an input to TopDown.

        Returns a Dict of the form {
            (county_1, enumdist_1) : pop_1,
            (county_1, enumdist_2) : pop_2,
                   ...
            (county_n, enumdist_x) : pop_k,
        }
    """
    enumdist_pops = dict()

    with open(input_file, "r") as raw_file:
        for line in raw_file:
            if line[0] == 'H' and line[53:55] == state_fips_code:
                # we have a household from this state!
                pop = int(line[15:17]) # pop of household
                county = int(line[55:59])
                enumdist = int(line[124:128])

                # add pop
                if (county, enumdist) in enumdist_pops.keys():
                    enumdist_pops[(county, enumdist)] += pop
                else:
                    enumdist_pops[(county, enumdist)] = pop

    return enumdist_pops

def pop_of_state_by_enumdist(input_file, state_fips_code):
    """ Returns a DataFrame with the populations of each enumdist in the state `state_fips_code`.
        `input_file` is the filepath of the .dat file used as input to TopDown.
    """
    enumdist_pops = pops_by_enumdist(input_file, state_fips_code)

    # populate a DataFrame with the enumdist populations
    main_df = pd.DataFrame(columns=["State", "County", "Enumdist", "TOTPOP"])

    for (county, enumdist) in enumdist_pops.keys():
        main_df = main_df.append({"State": int(state_fips_code),
                                  "County": county,
                                  "Enumdist": enumdist,
                                  "TOTPOP": enumdist_pops[(county, enumdist)]},
                                 ignore_index=True)
    return main_df
