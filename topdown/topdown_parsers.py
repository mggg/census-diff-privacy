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

def values_by_enumdist_for_run(df, state_id, race=None, race_percent=False, hisp=False, hisp_percent=False, vap=False):
    """ Generates and returns a DataFrame that tabulates "values" by enumdist
        for a single run.

        By default, this value is the total population of the enumdist.
        If `race` is passed, then the total population of that race by enumdist
        is computed.
        If `race_percent` is passed, the % of the race in each enumdist is
        returned.
    """
    assert(not (hisp and race)), "Hispanic Race counts/%s are currently not supported."

    # These values are from the Census 2018 E2E DAS Specs, page 10.
    NON_HISP_VAL = 1
    HISP_VAL = 2
    VOTING_AGE = 18
    NON_VOTING_AGE = 17

    state = df[df["TABBLKST"] == state_id]

    if hisp and hisp_percent:
        tot_pops = state.groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"]).size().reset_index()
        tot_pops.columns = ["TABBLKST", "TABBLKCOU", "ENUMDIST", "Run"]

        if vap:
            state["match"] = (state.CENHISP == HISP_VAL) & (state.QAGE == VOTING_AGE)
        else:
            state["match"] = (state.CENHISP == HISP_VAL)

        values_df = state.groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"])['match'].agg([('Run', 'sum')]).reset_index()
        values_df["Run"]  = values_df["Run"] / tot_pops["Run"]

    elif hisp:
        if vap:
            state["match"] = (state.CENHISP == HISP_VAL) & (state.QAGE == VOTING_AGE)
        else:
            state["match"] = (state.CENHISP == HISP_VAL)

        values_df = state.groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"])["match"].agg([('Run', 'sum')]).reset_index()

    elif race and race_percent:
        tot_pops = state.groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"]).size().reset_index()
        tot_pops.columns = ["TABBLKST", "TABBLKCOU", "ENUMDIST", "Run"]

        if vap:
            state['match'] = (state.CENRACE == race) & (state.CENHISP == NON_HISP_VAL) & (state.QAGE == VOTING_AGE)
        else:
            state['match'] = (state.CENRACE == race) & (state.CENHISP == NON_HISP_VAL)

        values_df = state.groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"])['match'].agg([('Run', 'sum')]).reset_index()
        values_df["Run"]  = values_df["Run"] / tot_pops["Run"]

    elif race:

        if vap:
            state['match'] = (state.CENRACE == race) & (state.CENHISP == NON_HISP_VAL) & (state.QAGE == VOTING_AGE)
        else:
            state['match'] = (state.CENRACE == race) & (state.CENHISP == NON_HISP_VAL)

        values_df = state.groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"])['match'].agg([('Run', 'sum')]).reset_index()

    else:
        if vap:
            values_df = state[state.QAGE == VOTING_AGE].groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"]).size().reset_index()
        else:
            values_df = state.groupby(["TABBLKST", "TABBLKCOU", "ENUMDIST"]).size().reset_index()

    values_df.columns = ["State", "County", "Enumdist", "Run"]
    return values_df

def build_county_file_from_topdown_output(dir_name,
                                          county_fips,
                                          person_filename,
                                          county_filename):
    """
    """
    for root, dirs, files in os.walk(dir_name):
        for d in dirs:
            if d[:7] == "output_":
                path = os.path.join(root, d)
                person_file = path + "/" + person_filename
                out_file = path + "/" + county_filename

                with open(person_file, "r") as input_file, open(out_file, "w") as write_file:
                    for line in input_file:
                        if line[17:20] == county_fips:
                            write_file.write(line)
                print("Saved County at {}".format(out_file))


def collect_by_enumdist(dir_name,
                        state_id,
                        filename,
                        race=None,
                        race_percent=False,
                        hisp=False,
                        hisp_percent=False,
                        vap=False):
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
                person_file = path + "/" + filename

                person_df = read_df_1940(person_file)
                values_df = values_by_enumdist_for_run(person_df,
                                                       state_id,
                                                       race,
                                                       race_percent,
                                                       hisp,
                                                       hisp_percent,
                                                       vap)

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

def distribution_of_input_var(input_file, line_type,
                              single_idx=None, start_idx=None, end_idx=None):
    """ Returns all the values at columns `single_idx` or (`start_idx, `end_idx`)
        of each line of `line_type` of `input_file` as a list.
        `line_type` denotes whether the line is a Person line or a Household Line.

        The indexes should be entered **as is** as appears on the count codebook here:
        https://usa.ipums.org/usa/resources/1940CensusDASTestData/EXT1940USCB.cbk

        Example Usage:
            distribution_of_input_var(file, "H", single_idx=17)
            distribution_of_input_var(file, "H", start_idx=2, end_idx=5)
    """
    # verify inputs
    assert(line_type in ["P", "H"])
    assert(single_idx != 0)
    if single_idx:
        assert(start_idx == None and end_idx==None)
    else:
        assert(start_idx)
        assert(end_idx)

    all_vars = []
    with open(input_file, "r") as raw_file:
        for line in raw_file:
            if line[0] == line_type:
                if single_idx:
                    all_vars.append(line[single_idx-1])
                else:
                    all_vars.append(line[start_idx-1:end_idx])
    return all_vars

###################################################################################
###################################################################################
###################################################################################


def parse_reconstructed_geo_output(df, geo_col="NAME"):
    """ Parses the `geo_col` column of the reconstructions in to block, block group, tract, county and state columns.
    """
    df[["block", "bg", "tract", "county", "state"]] = df[geo_col].str.split(", ", expand=True)
    df["block"] = df["block"].str.split(expand=True)[1]
    df["bg"] = df["bg"].str.split(expand=True)[2]
    df["tract"] = df["tract"].str.split(expand=True)[2]
    df["tract"] = df["tract"].str.replace(".","").str.pad(width=6, side='left', fillchar='0')
    df["county"] = df["county"].str.split(expand=True).iloc[:,:-1].apply(lambda x: ' '.join(x), axis=1)
    return df

def build_enumdist_col(df):
    """ Concatenates the tract, bg (block group) and block columns of `df` to produce an "enumdist" column.
        Returns the new dataframe.
    """
    df["enumdist"] = df[["tract", "bg", "block"]].apply(lambda x: ''.join(x), axis=1)
    return df

def get_sample_1940_hh():
    """ Returns a sample Household line of 1940 ipums format. This line was randomly taken from the 1940 alabama.
    """
    hh_line = "H19400200024278096700000001000009100000000001198632410100102100000009999000260300026007000840199990012200020999999901223233100110101000000001000900000000100090"
    return hh_line

def get_sample_1940_person():
    """ Returns a sample Person line of 1940 ipums format. This line was randomly taken from the 1940 alabama.
    """
    person_line = "P19400200024278000900000001000000000000110109213070306030000019999090901101499600000110000000010010003703700018018000000000010212120030303331099599909950000000000009999999999990000000100000009999999999991109909999199990072199990000000A59B1CD2-5F9A-47AB-AF36-E5F4D7F65F0B020"
    return person_line

def parse_positions_hh(line):
    """ Parse positions of Household lines for the input files and return it as a dictionary.
    """
    dictionary = dict()
    dictionary["RECTYPE"] = line[0:1]
    dictionary["YEAR"] = line[1:5]
    dictionary["DATANUM"] = line[5:7]
    dictionary["SERIAL"] = line[7:15]
    dictionary["NUMPREC"] = line[15:17]
    dictionary["SUBSAMP"] = line[17:19]
    dictionary["HHWT"] = line[19:29]
    dictionary["NUMPERHH"] = line[29:33]
    dictionary["HHTYPE"] = line[33:34]
    dictionary["DWELLING"] = line[34:42]
    dictionary["SLPERNUM"] = line[42:44]
    dictionary["CPI99"] = line[44:49]
    dictionary["REGION"] = line[49:51]
    dictionary["STATEICP"] = line[51:53]
    dictionary["STATEFIP"] = line[53:55]
    dictionary["COUNTY"] = line[55:59]
    dictionary["URBAN"] = line[59:60]
    dictionary["METRO"] = line[60:61]
    dictionary["METAREA"] = line[61:64]
    dictionary["METAREAD"] = line[64:68]
    dictionary["CITY"] = line[68:72]
    dictionary["CITYPOP"] = line[72:77]
    dictionary["SIZEPL"] = line[77:79]
    dictionary["URBPOP"] = line[79:84]
    dictionary["SEA"] = line[84:87]
    dictionary["WARD"] = line[87:90]
    dictionary["CNTRY"] = line[90:93]
    dictionary["GQ"] = line[93:94]
    dictionary["GQTYPE"] = line[94:95]
    dictionary["GQTYPED"] = line[95:98]
    dictionary["GQFUNDS"] = line[98:100]
    dictionary["FARM"] = line[100:101]
    dictionary["OWNERSHP"] = line[101:102]
    dictionary["OWNERSHPD"] = line[102:104]
    dictionary["RENT"] = line[104:108]
    dictionary["VALUEH"] = line[108:115]
    dictionary["NFAMS"] = line[115:117]
    dictionary["NSUBFAM"] = line[117:118]
    dictionary["NCOUPLES"] = line[118:119]
    dictionary["NMOTHERS"] = line[119:120]
    dictionary["NFATHERS"] = line[120:121]
    dictionary["MULTGEN"] = line[121:122]
    dictionary["MULTGEND"] = line[122:124]
    dictionary["ENUMDIST"] = line[124:128]
    dictionary["SUPDIST"] = line[128:131]
    dictionary["RESPOND"] = line[131:132]
    dictionary["SPLIT"] = line[132:133]
    dictionary["SPLITHID"] = line[133:141]
    dictionary["SPLITNUM"] = line[141:145]
    dictionary["SPLIT40"] = line[145:146]
    dictionary["SERIAL40"] = line[146:154]
    dictionary["NUMPREC40"] = line[154:158]
    dictionary["EDMISS"] = line[158:159]

    return dictionary

def parse_positions_person(line):
    """ Parse positions of Person lines for the input files and return it as a dictionary.
    """
    dictionary = dict()
    dictionary["RECTYPE"] = line[0:1]
    dictionary["YEAR"] = line[1:5]
    dictionary["DATANUM"] = line[5:7]
    dictionary["SERIAL"] = line[7:15]
    dictionary["PERNUM"] = line[15:19]
    dictionary["PERWT"] = line[19:29]
    dictionary["SLWT"] = line[29:39]
    dictionary["SLREC"] = line[39:40]
    dictionary["RESPONDT"] = line[40:41]
    dictionary["FAMUNIT"] = line[41:43]
    dictionary["FAMSIZE"] = line[43:45]
    dictionary["SUBFAM"] = line[45:46]
    dictionary["SFTYPE"] = line[46:47]
    dictionary["SFRELATE"] = line[47:48]
    dictionary["MOMLOC"] = line[48:50]
    dictionary["STEPMOM"] = line[50:51]
    dictionary["MOMRULE_HIST"] = line[51:52]
    dictionary["POPLOC"] = line[52:54]
    dictionary["STEPPOP"] = line[54:55]
    dictionary["POPRULE_HIST"] = line[55:56]
    dictionary["SPLOC"] = line[56:58]
    dictionary["SPRULE_HIST"] = line[58:59]
    dictionary["NCHILD"] = line[59:60]
    dictionary["NCHLT5"] = line[60:61]
    dictionary["NSIBS"] = line[61:62]
    dictionary["ELDCH"] = line[62:64]
    dictionary["YNGCH"] = line[64:66]
    dictionary["RELATE"] = line[66:68]
    dictionary["RELATED"] = line[68:72]
    dictionary["SEX"] = line[72:73]
    dictionary["AGE"] = line[73:76]
    dictionary["AGEMONTH"] = line[76:78]
    dictionary["MARST"] = line[78:79]
    dictionary["MARRNO"] = line[79:80]
    dictionary["AGEMARR"] = line[80:82]
    dictionary["CHBORN"] = line[82:84]
    dictionary["RACE"] = line[84:85]
    dictionary["RACED"] = line[85:88]
    dictionary["HISPAN"] = line[88:89]
    dictionary["HISPAND"] = line[89:92]
    dictionary["BPL"] = line[92:95]
    dictionary["BPLD"] = line[95:100]
    dictionary["MBPL"] = line[100:103]
    dictionary["MBPLD"] = line[103:108]
    dictionary["FBPL"] = line[108:111]
    dictionary["FBPLD"] = line[111:116]
    dictionary["NATIVITY"] = line[116:117]
    dictionary["CITIZEN"] = line[117:118]
    dictionary["MTONGUE"] = line[118:120]
    dictionary["MTONGUED"] = line[120:124]
    dictionary["SPANNAME"] = line[124:125]
    dictionary["HISPRULE"] = line[125:126]
    dictionary["SCHOOL"] = line[126:127]
    dictionary["HIGRADE"] = line[127:129]
    dictionary["HIGRADED"] = line[129:132]
    dictionary["EDUC"] = line[132:134]
    dictionary["EDUCD"] = line[134:137]
    dictionary["EMPSTAT"] = line[137:138]
    dictionary["EMPSTATD"] = line[138:140]
    dictionary["LABFORCE"] = line[140:141]
    dictionary["OCC"] = line[141:145]
    dictionary["OCC1950"] = line[145:148]
    dictionary["IND"] = line[148:152]
    dictionary["IND1950"] = line[152:155]
    dictionary["CLASSWKR"] = line[155:156]
    dictionary["CLASSWKRD"] = line[156:158]
    dictionary["WKSWORK1"] = line[158:160]
    dictionary["WKSWORK2"] = line[160:161]
    dictionary["HRSWORK1"] = line[161:163]
    dictionary["HRSWORK2"] = line[163:164]
    dictionary["DURUNEMP"] = line[164:167]
    dictionary["UOCC"] = line[167:170]
    dictionary["UOCC95"] = line[170:173]
    dictionary["UIND"] = line[173:176]
    dictionary["UCLASSWK"] = line[176:177]
    dictionary["INCWAGE"] = line[177:183]
    dictionary["INCNONWG"] = line[183:184]
    dictionary["OCCSCORE"] = line[184:186]
    dictionary["SEI"] = line[186:188]
    dictionary["PRESGL"] = line[188:191]
    dictionary["ERSCOR50"] = line[191:195]
    dictionary["EDSCOR50"] = line[195:199]
    dictionary["NPBOSS50"] = line[199:203]
    dictionary["MIGRATE5"] = line[203:204]
    dictionary["MIGRATE5D"] = line[204:206]
    dictionary["MIGPLAC5"] = line[206:209]
    dictionary["MIGMET5"] = line[209:213]
    dictionary["MIGTYPE5"] = line[213:214]
    dictionary["MIGCITY5"] = line[214:218]
    dictionary["MIGSEA5"] = line[218:221]
    dictionary["SAMEPLAC"] = line[221:222]
    dictionary["SAMESEA5"] = line[222:223]
    dictionary["MIGCOUNTY"] = line[223:227]
    dictionary["VETSTAT"] = line[227:228]
    dictionary["VETSTATD"] = line[228:230]
    dictionary["VET1940"] = line[230:231]
    dictionary["VETWWI"] = line[231:232]
    dictionary["VETPER"] = line[232:233]
    dictionary["VETCHILD"] = line[233:234]
    dictionary["HISTID"] = line[234:270]
    dictionary["SURSIM"] = line[270:272]
    dictionary["SSENROLL"] = line[272:273]
    return dictionary

def left_pad_with_zeros(number, target_len):
    """ Left-pads a number with 0s until the string gets to length `target_len`.
        Eg. if number is 123 and target_len = 5, the return value is `00123`.

        Note: `number` doesn't necessarily have to be a number for the function to work.
    """
    str_num = str(number)
    assert(len(str_num) <= target_len)

    zeros_needed = target_len - len(str_num)
    prefix = ''
    for i in range(zeros_needed):
        prefix += '0'

    return prefix + str_num

def modify_age(line, age, age_len=3, age_col="AGE"):
    """ Changes the age value of the dictionary `line`,
        and returns the new updated dictionary.
    """
    line[age_col] = left_pad_with_zeros(str(age), age_len)
    return line

def modify_race(line, race, race_col="RACE"):
    """ Changes the `race` field of the dict `line`.
        The 2018 E2E can only take 6 fields as input [1-6], and this is a
        mapping of the reconstructed races to __arbitrary__ input values.

        Importantly, multi-racial people have been labelled as other in this convention.
    """
    if race == "w":
        line[race_col] = '1'
    elif race == "b":
        line[race_col] = '2'
    elif race == "i":
        line[race_col] = '3'
    elif race == "a":
        line[race_col] = '4'
    elif race == "h":
        line[race_col] = '5'
    elif race == "o":
        line[race_col] = '6'
    elif len(race) >= 2:  # multi-racial people are labeled as other.
        line[race_col] = '6'
    else:
        raise Exception("Race not in [w, b, i, a, h, o]: {}".format(race))

    return line

def modify_hisp(line, hisp, hisp_col="HISPAN"):
    """ Changes the `hisp_col` value of the dictionary `line` to `hisp`,
        and returns the new updated dictionary.
    """
    assert(str(hisp) in ['0', '1'])
    line[hisp_col] = str(hisp)
    return line

def modify_gqtype(line, gqtype, gqtype_col="GQTYPE"):
    """ Changes the `gqtype_col` value of the dictionary `line` to `gqtype`,
        and returns the new updated dictionary.
    """
    assert(len(str(gqtype)) == 1)
    line[gqtype_col] = str(gqtype)
    return line

def modify_gq(line, gq, gq_col="GQ"):
    """ Changes the `gq_col` value of the dictionary `line` to `gq`,
        and returns the new updated dictionary.
    """
    assert(len(str(gq))==1)
    line[gq_col] = str(gq)
    return line

def modify_serial(line, serial, serial_len=8, serial_col="SERIAL"):
    """ Changes the `serial_col` value of the dictionary `line` to `serial`,
        and returns the new updated dictionary.
    """
    line[serial_col] = left_pad_with_zeros(serial, serial_len)
    return line

def modify_state(line, state, state_col="STATEFIP", state_len=2):
    """ Changes the `state_col` value of the dictionary `line` to `state`,
        and returns the new updated dictionary.
    """
    line[state_col] = left_pad_with_zeros(state, state_len)
    return line

def modify_county(line, county, county_col="COUNTY", county_len=3):
    """ Changes the `county_col` value of the dictionary `line` to `county`,
        and returns the new updated dictionary.
    """
    line[county_col] = left_pad_with_zeros(county, county_len)
    return line

def modify_enumdist(line, enumdist, enumdist_col="ENUMDIST"):
    """ Changes the `enumdist_col` value of the dictionary `line` to `enumdist`,
        and returns the new updated dictionary.
    """
    line[enumdist_col] = enumdist
    return line

def get_texas_county_fips_code_map():
    """ Returns a dictionary that has the name of Texas counties as keys and their corresponding
        county fips codes as values.
    """
    county_names = [
        "Anderson",
        "Andrews",
        "Angelina",
        "Aransas",
        "Archer",
        "Armstrong",
        "Atascosa",
        "Austin",
        "Bailey",
        "Bandera",
        "Bastrop",
        "Baylor",
        "Bee",
        "Bell",
        "Bexar",
        "Blanco",
        "Borden",
        "Bosque",
        "Bowie",
        "Brazoria",
        "Brazos",
        "Brewster",
        "Briscoe",
        "Brooks",
        "Brown",
        "Burleson",
        "Burnet",
        "Caldwell",
        "Calhoun",
        "Callahan",
        "Cameron",
        "Camp",
        "Carson",
        "Cass",
        "Castro",
        "Chambers",
        "Cherokee",
        "Childress",
        "Clay",
        "Cochran",
        "Coke",
        "Coleman",
        "Collin",
        "Collingsworth",
        "Colorado",
        "Comal",
        "Comanche",
        "Concho",
        "Cooke",
        "Coryell",
        "Cottle",
        "Crane",
        "Crockett",
        "Crosby",
        "Culberson",
        "Dallam",
        "Dallas",
        "Dawson",
        "Deaf Smith",
        "Delta",
        "Denton",
        "DeWitt",
        "Dickens",
        "Dimmit",
        "Donley",
        "Duval",
        "Eastland",
        "Ector",
        "Edwards",
        "Ellis",
        "El Paso",
        "Erath",
        "Falls",
        "Fannin",
        "Fayette",
        "Fisher",
        "Floyd",
        "Foard",
        "Fort Bend",
        "Franklin",
        "Freestone",
        "Frio",
        "Gaines",
        "Galveston",
        "Garza",
        "Gillespie",
        "Glasscock",
        "Goliad",
        "Gonzales",
        "Gray",
        "Grayson",
        "Gregg",
        "Grimes",
        "Guadalupe",
        "Hale",
        "Hall",
        "Hamilton",
        "Hansford",
        "Hardeman",
        "Hardin",
        "Harris",
        "Harrison",
        "Hartley",
        "Haskell",
        "Hays",
        "Hemphill",
        "Henderson",
        "Hidalgo",
        "Hill",
        "Hockley",
        "Hood",
        "Hopkins",
        "Houston",
        "Howard",
        "Hudspeth",
        "Hunt",
        "Hutchinson",
        "Irion",
        "Jack",
        "Jackson",
        "Jasper",
        "Jeff Davis",
        "Jefferson",
        "Jim Hogg",
        "Jim Wells",
        "Johnson",
        "Jones",
        "Karnes",
        "Kaufman",
        "Kendall",
        "Kenedy",
        "Kent",
        "Kerr",
        "Kimble",
        "King",
        "Kinney",
        "Kleberg",
        "Knoxv",
        "Lamar",
        "Lamb",
        "Lampasas",
        "La Salle",
        "Lavaca",
        "Lee",
        "Leon",
        "Liberty",
        "Limestone",
        "Lipscomb",
        "Live Oak",
        "Llano",
        "Loving",
        "Lubbock",
        "Lynn",
        "McCulloch",
        "McLennan",
        "McMullen",
        "Madison",
        "Marion",
        "Martin",
        "Mason",
        "Matagorda",
        "Maverick",
        "Medina",
        "Menard",
        "Midland",
        "Milam",
        "Mills",
        "Mitchell",
        "Montague",
        "Montgomery",
        "Moore",
        "Morris",
        "Motley",
        "Nacogdoches",
        "Navarro",
        "Newton",
        "Nolan",
        "Nueces",
        "Ochiltree",
        "Oldham",
        "Orange",
        "Palo Pinto",
        "Panola",
        "Parker",
        "Parmer",
        "Pecos",
        "Polk",
        "Potter",
        "Presidio",
        "Rains",
        "Randall",
        "Reagan",
        "Real",
        "Red River",
        "Reeves",
        "Refugio",
        "Roberts",
        "Robertson",
        "Rockwall",
        "Runnels",
        "Rusk",
        "Sabine",
        "San Augustine",
        "San Jacinto",
        "San Patricio",
        "San Saba",
        "Schleicher",
        "Scurry",
        "Shackelford",
        "Shelby",
        "Sherman",
        "Smith",
        "Somervell",
        "Starr",
        "Stephens",
        "Sterling",
        "Stonewall",
        "Sutton",
        "Swisher",
        "Tarrant",
        "Taylor",
        "Terrell",
        "Terry",
        "Throckmorton",
        "Titus",
        "Tom Green",
        "Travis",
        "Trinity",
        "Tyler",
        "Upshur",
        "Upton",
        "Uvalde",
        "Val Verde",
        "Van Zandt",
        "Victoria",
        "Walker",
        "Waller",
        "Ward",
        "Washington",
        "Webb",
        "Wharton",
        "Wheeler",
        "Wichita",
        "Wilbarger",
        "Willacy",
        "Williamson",
        "Wilson",
        "Winkler",
        "Wise",
        "Wood",
        "Yoakum",
        "Young",
        "Zapata",
        "Zavala"
    ]

    fips_code_map = dict()

    counter = 1
    for county in county_names:
        fips_code_map[county] = left_pad_with_zeros(counter, 3)
        counter += 2

    return fips_code_map

def convert_to_hh_line_delimited(hh):
    """ Takes in the dictionary `hh` and returns a pipe-delimited line with the values in the order
        of the `hh_fields` list below.
    """
    hh_fields = ['RECTYPE', 'YEAR', 'DATANUM', 'SERIAL', 'NUMPREC', 'SUBSAMP',
                 'HHWT', 'NUMPERHH', 'HHTYPE', 'DWELLING', 'SLPERNUM', 'CPI99',
                 'REGION', 'STATEICP', 'STATEFIP', 'COUNTY', 'URBAN', 'METRO',
                 'METAREA', 'METAREAD', 'CITY', 'CITYPOP', 'SIZEPL', 'URBPOP',
                 'SEA', 'WARD', 'CNTRY', 'GQ', 'GQTYPE', 'GQTYPED', 'GQFUNDS',
                 'FARM', 'OWNERSHP', 'OWNERSHPD', 'RENT', 'VALUEH', 'NFAMS',
                 'NSUBFAM', 'NCOUPLES', 'NMOTHERS', 'NFATHERS', 'MULTGEN',
                 'MULTGEND', 'ENUMDIST', 'SUPDIST', 'RESPOND', 'SPLIT', 'SPLITHID',
                 'SPLITNUM', 'SPLIT40', 'SERIAL40', 'NUMPREC40', 'EDMISS']

    line_list = []
    for field in hh_fields:
        line_list.append(hh[field])

    # append a new line at the end
    # line_list.append("\n")

    line = '|'.join(line_list)
    line = line + "\n"
    return line


def convert_to_person_line_delimited(person):
    """ Takes in the dictionary `person` and returns a pipe-delimited line with the values in the order
        of the `person_fields` list below.
    """
    person_fields = ['RECTYPE', 'YEAR', 'DATANUM', 'SERIAL', 'PERNUM', 'PERWT',
                     'SLWT', 'SLREC', 'RESPONDT', 'FAMUNIT', 'FAMSIZE', 'SUBFAM',
                     'SFTYPE', 'SFRELATE', 'MOMLOC', 'STEPMOM', 'MOMRULE_HIST',
                     'POPLOC', 'STEPPOP', 'POPRULE_HIST', 'SPLOC', 'SPRULE_HIST',
                     'NCHILD', 'NCHLT5', 'NSIBS', 'ELDCH', 'YNGCH', 'RELATE',
                     'RELATED', 'SEX', 'AGE', 'AGEMONTH', 'MARST', 'MARRNO',
                     'AGEMARR', 'CHBORN', 'RACE', 'RACED', 'HISPAN', 'HISPAND',
                     'BPL', 'BPLD', 'MBPL', 'MBPLD', 'FBPL', 'FBPLD', 'NATIVITY',
                     'CITIZEN', 'MTONGUE', 'MTONGUED', 'SPANNAME', 'HISPRULE',
                     'SCHOOL', 'HIGRADE', 'HIGRADED', 'EDUC', 'EDUCD', 'EMPSTAT',
                     'EMPSTATD', 'LABFORCE', 'OCC', 'OCC1950', 'IND', 'IND1950',
                     'CLASSWKR', 'CLASSWKRD', 'WKSWORK1', 'WKSWORK2', 'HRSWORK1',
                     'HRSWORK2', 'DURUNEMP', 'UOCC', 'UOCC95', 'UIND', 'UCLASSWK',
                     'INCWAGE', 'INCNONWG', 'OCCSCORE', 'SEI', 'PRESGL', 'ERSCOR50',
                     'EDSCOR50', 'NPBOSS50', 'MIGRATE5', 'MIGRATE5D', 'MIGPLAC5',
                     'MIGMET5', 'MIGTYPE5', 'MIGCITY5', 'MIGSEA5', 'SAMEPLAC',
                     'SAMESEA5', 'MIGCOUNTY', 'VETSTAT', 'VETSTATD', 'VET1940',
                     'VETWWI', 'VETPER', 'VETCHILD', 'HISTID', 'SURSIM', 'SSENROLL']

    line_list = []
    for field in person_fields:
        line_list.append(person[field])

    # append a new line at the end
    # line_list.append("\n")

    line = '|'.join(line_list)
    line = line + "\n"
    return line

def build_person_line(serial, age, hisp, race, serial_len=8):
    """ Generates a Person line in 1940s ipums format and changes its
        `serial`, `age`, `hisp` and `race`.
        Returns the line.
    """
    person_line = get_sample_1940_person()
    person = parse_positions_person(person_line)
    person = modify_serial(person, serial, serial_len=serial_len)
    person = modify_age(person, age)
    person = modify_hisp(person, hisp)
    person = modify_race(person, race)

    person_line = convert_to_person_line_delimited(person)
    return person_line

def build_hh_line(serial, gq, gqtype, state, county, enumdist, serial_len=8):
    """ Generates a Household line in 1940s ipums format and changes its
        `serial`, `gq`, `gqtype`, `state`, `county` and `enudmist`.
        Returns the line.
    """
    sample_line = get_sample_1940_hh()
    hh = parse_positions_hh(sample_line)

    hh = modify_serial(hh, serial, serial_len=serial_len)
    hh = modify_gq(hh, gq)
    hh = modify_gqtype(hh, gqtype)
    hh = modify_state(hh, state)
    hh = modify_county(hh, county)
    hh = modify_enumdist(hh, enumdist)
    hh_line = convert_to_hh_line_delimited(hh)
    return hh_line

def write_household_to_file(write_file, hh_line, person_lines):
    """ Writes `hh_line` and all the person lines in the list `person_lines`
        to the open file object `write_file`.
        The `hh_line` is written first, and then the `person_lines` are written.
        The ideas is that all the people in person lines are in the same household.
    """
    write_file.write(hh_line)
    for person_line in person_lines:
        write_file.write(person_line)

fips_map = get_texas_county_fips_code_map()

def county_fips(county):
    """ Given a Texas county name (eg. "Dallas"), returns a 3 char fips code
        of the county.
    """
    return fips_map[county]

def state_fips(state):
    """ Given a State name (eg. "Texas"), returns a 2 char fips code.
        Hacky: currently only supports Texas lol.
    """
    if state == "Texas":
        return '48'

def read_and_process_reconstructed_csvs(dir_name):
    """ Reads all the .csvs in filepath `dir_name` and concats them
        into one dataframe.
        Proceeds to post-process the data frame to seperate state, county,
        tract, block group and block. The tract, block group and block are
        concatenated together to form an `enumdist`.
        Also converts the state and county to their fips codes.

        Returns this df.
    """
    print("Reading files...")
    df = read_reconstructions(dir_name)
    df = parse_reconstructed_geo_output(df)
    df = build_enumdist_col(df)

    df["state"] = df["state"].apply(lambda state: state_fips(state))
    df["county"] = df["county"].apply(lambda county: county_fips(county))
    print("Completed reading files")

    return df

def read_reconstructions(dir_name):
    """ Reads all the .csvs in filepath `dir_name` and concats them
        into one dataframe.
        Returns this dataframe.
    """
    main_df = pd.DataFrame()
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file[-3:] != "csv":
                continue

            curr_df = pd.read_csv(os.path.join(root, file))
            curr_df = parse_reconstructed_geo_output(curr_df)
            curr_df = build_enumdist_col(curr_df)

            # duplicate the rows based on the column `sol` i.e if sol = 3 that row is
            # converted into 3 rows.
            curr_df = pd.DataFrame([curr_df.loc[idx]
                                    for idx in curr_df.index
                                    for _ in range(curr_df.loc[idx]['sol'])]).reset_index(drop=True)

            main_df = pd.concat([main_df, curr_df])

    return main_df


def convert_reconstructions_to_ipums(dir_name,
                                     save_fp,
                                     hh_size=5,
                                     gq=1,
                                     gqtype=0,
                                     break_size=500):
    """ Converts all the .csvs in `dir_name` into ipums format lines and saves the file to `save_fp`.

        Other arguments:
            hh_size (int): Size of households the person lines by block are grouped into.
                           (eg. if a block has 12 people and hh_size = 5,
                            three households of size 5, 5, and 2 are created.)
            gq (int): Group Quarters code to be added to all the household lines.
                      A default value of 1 means that all the households are
                      regular households (as opposed to group quarters)
            gqtype(int): Group Quarters code to be added to all the household lines.
                      A default value of 0 means that all the households are
                      regular households (as opposed to say colleges or jails)
            break_size(int): Number of blocks to write before a print() statement updates
                      on how far along the conversion has gone. A progress bar of sorts.
    """
    # read the files, and process them
    df = read_and_process_reconstructed_csvs(dir_name)

    print("Grouping the data at a block level...")
    groups = df.groupby(["state", "county", "tract", "bg", "block"])
    print("Finished grouping the data.")
    total_written = 0
    counter = 0 # counter at a block level

    with open(save_fp, "w+") as write_file:
        serial = 1
        for ((state, county, tract, bg, block), group) in groups:

            counter += 1
            if counter % break_size == 0:
                print("Writing block {} of {}.".format(counter, len(groups)))

            block_df = df[(df["state"]==state) & (df["county"]==county) & (df["tract"]==tract) & (df["bg"]==bg) & (df["block"]==block)]
            person_lines = []

            for (_, row) in block_df.iterrows():
                person_line = build_person_line(serial, row["age"], row["ethn"], row["race"])
                person_lines.append(person_line)

                if (len(person_lines) == hh_size):
                    hh_line = build_hh_line(serial, gq, gqtype, state, county, row["enumdist"])
                    write_household_to_file(write_file, hh_line, person_lines)
                    serial += 1
                    person_lines = []

            if len(person_lines) > 0:
                # scoop up the remaining lines that are not % hh_size == 0
                hh_line = build_hh_line(serial, gq, gqtype, state, county, row["enumdist"])
                write_household_to_file(write_file, hh_line, person_lines)
                serial += 1
                person_lines = []

def convert_reconstructions_to_ipums_same_block(dir_name,
                                                save_fp,
                                                hh_size=5,
                                                gq=1,
                                                gqtype=0,
                                                break_size=500):
    """
    Converts all the .csvs in `dir_name` into ipums format lines and saves the file to `save_fp`.
    Differs from `convert_reconstructions_to_ipums()` in that it puts all the people in `dir_name`
    IN THE SAME BLOCK, ie it ignores the block, block group and tract assignments from the reconstructions
    and puts the people into the same block "0001".
        Other arguments:
            hh_size (int): Size of households the person lines by block are grouped into.
                           (eg. if a block has 12 people and hh_size = 5,
                            three households of size 5, 5, and 2 are created.)
            serial (int) : Serial number that the reconstruction lines starts with.
                           Everyone in a household has the same serial number, and the
                           household also contains the serial number. Each household has
                           a unique serial number.
            gq (int): Group Quarters code to be added to all the household lines.
                      A default value of 1 means that all the households are
                      regular households (as opposed to group quarters)
            gqtype(int): Group Quarters code to be added to all the household lines.
                      A default value of 0 means that all the households are
                      regular households (as opposed to say colleges or jails)
            break_size(int): Number of blocks to write before a print() statement updates
                      on how far along the conversion has gone. A progress bar of sorts.
        Returns the serial number that is (last serial number used for this dir) + 1
        i.e this return value can safely be used as a serial number for other reconstructions outside this function.
        Also returns the number of people reconstructed.
    """
    # read the files, and process them
    df = read_and_process_reconstructed_csvs(dir_name)
    df["enumdist"] = "99999999999"
    df = df.reset_index()

    with open(save_fp, "w+") as write_file:
        # first serial in geoid. 7 digits because a county's pop can go up to single digit millions.
        serial = int( df["state"].iloc[0]
                    + df["county"].iloc[0]
                    + df["enumdist"].iloc[0]
                    + "0000001")

        person_lines = []

        for (idx, row) in df.iterrows():

            if idx % break_size == 0:
                print("Writing person {} of {}.".format(idx, len(df)))

            person_line = build_person_line(serial, row["age"], row["ethn"], row["race"], serial_len=len(str(serial)))
            person_lines.append(person_line)

            if (len(person_lines) == hh_size):
                hh_line = build_hh_line(serial, gq, gqtype, row["state"], row["county"], row["enumdist"], serial_len=len(str(serial)))
                write_household_to_file(write_file, hh_line, person_lines)
                serial += 1
                person_lines = []

        if len(person_lines) > 0:
            # scoop up the remaining lines that are not % hh_size == 0
            hh_line = build_hh_line(serial, gq, gqtype, row["state"], row["county"], row["enumdist"], serial_len=len(str(serial)))
            write_household_to_file(write_file, hh_line, person_lines)
            serial += 1
            person_lines = []
