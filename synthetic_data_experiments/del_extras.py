dir_name = "./NORMAL_CONFIG_23/"

import os

for root, dirs, files in os.walk(dir_name):
    for d in dirs:
        if d[:7] == "output_":
            person_dir = os.path.join(root, d) + "/MDF_PER.txt"
            unit_dir = os.path.join(root, d) + "/MDF_UNIT.txt"
            zip_file = os.path.join(root, d) + "/MDF_RESULTS.zip"
            os.system("rm -rf {}".format(person_dir))
            os.system("rm -rf {}".format(unit_dir))
            os.system("rm {}".format(zip_file))
