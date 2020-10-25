import pandas as pd
import os

## Input Variables
input_file = "alabama_1940_raw.dat"
reduced_file = "reduced_file.dat"
output_file = "normal_experiment_input.dat"

races = ["1", "2"]

selected_counties = [10, 30, 50]

county_dists = dict({10: [11, 12, 20],
                     30: [10, 20, 30],
                     50: [10, 20, 30]})

# to hold the target count of the district
dist_pops = dict({(10, 11) : 345,
                  (10, 12) : 366,
                  (10, 20) : 260,
                  (30, 10) : 289,
                  (30, 20) : 294,
                  (30, 30) : 279,
                  (50, 10) : 200,
                  (50, 20) : 211,
                  (50, 30) : 151,
                 })

# populations of race A
race_pops = dict({(10, 11) : 60,
                  (10, 12) : 101,
                  (10, 20) : 102,
                  (30, 10) : 112,
                  (30, 20) : 100,
                  (30, 30) : 116,
                  (50, 10) : 120,
                  (50, 20) : 161,
                  (50, 30) : 138,
                 })

##

def initialize_counts_dict(county_dists):
    """
    """
    empty_counts = dict()
    for county in county_dists.keys():
        for dist in county_dists[county]:
            empty_counts[(county, dist)] = 0
    return empty_counts

def build_file_with_specified_population(input_file, output_file, selected_counties, county_dists, dist_pops):
    """
    """
    dist_counts = initialize_counts_dict(county_dists)

    with open(input_file, "r") as raw_file, open(output_file, "w") as out_file:
        for line in raw_file:
            if line[0] == 'H':
                writeable = False

                hh_pop = int(line[15:17]) # population of household
                county = int(line[55:59])
                enumdist = int(line[124:128])

                if county not in selected_counties or enumdist not in county_dists[county]:
                    continue

                if hh_pop + dist_counts[(county, enumdist)] > dist_pops[county, enumdist]:
                    continue
                else:
                    out_file.write(line)
                    dist_counts[(county, enumdist)] += hh_pop
                    writeable = True

            elif line[0] == "P" and writeable:
                out_file.write(line)

def num_persons_in_file(filename):
    """
    """
    pop = 0
    with open(filename, "r") as input_file:
        for line in input_file:
            if line[0] == "P":
                pop += 1
    return pop

def num_hhs_in_file(filename):
    """
    """
    hhs = 0
    with open(filename, "r") as input_file:
        for line in input_file:
            if line[0] == "H":
                hhs += 1
    return hhs

def assign_race_to_people(input_file, output_file, races, county_dists, race_pops):
    """
    """
    dist_counts = initialize_counts_dict(county_dists)
    assert(len(races) == 2)

    with open(input_file, "r") as input_file, open(output_file, "w") as out_file:
        for line in input_file:
            if line[0] == 'H':
                county = int(line[55:59])
                enumdist = int(line[124:128])
                out_file.write(line)

            elif line[0] == "P":
                if dist_counts[(county, enumdist)] < race_pops[(county, enumdist)]:
                    new_line = line[:84] + races[0] + line[85:]
                else:
                    new_line = line[:84] + races[1] + line[85:]

                assert(len(line) == len(new_line))

                out_file.write(new_line)
                dist_counts[(county, enumdist)] += 1

def counts_by_race(file, races):
    """
    """
    assert(len(races) == 2)

    race_a = 0
    race_b = 0
    with open(file, "r") as input_file:
        for line in input_file:
            if line[0] == 'P':
                if line[84] == races[0]:
                    race_a += 1
                elif line[84] == races[1]:
                    race_b += 1
    return race_a, race_b

def validate_num_people_in_household(file):
    ""
    ""
    with open(file, "r") as input_file:
        num_people_in_hh = 0
        hh_pop = 0
        for line in input_file:
            if line[0] == 'H':
                assert(hh_pop == num_people_in_hh)
                hh_pop = int(line[15:17])
                num_people_in_hh = 0
            elif line[0] == "P":
                num_people_in_hh += 1


# build the ipums format file with specified population, then assign races
build_file_with_specified_population(input_file, reduced_file, selected_counties, county_dists, dist_pops)
assign_race_to_people(reduced_file, output_file, ["1", "2"], county_dists, race_pops)

# tests
race_a, race_b = counts_by_race(output_file, races)
assert(num_persons_in_file(output_file) == sum(dist_pops.values()))
assert(race_a == sum(race_pops.values()))
assert(race_b == sum(dist_pops.values()) - race_a)
assert(num_hhs_in_file(output_file) > 0)
validate_num_people_in_household(output_file)

# Done! Cleanups
print("All tests have passed!")
os.remove(reduced_file)
