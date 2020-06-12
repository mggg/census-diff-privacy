
dir_names = ["./NORMAL_CONFIG_31.ini/",
             "./NORMAL_CONFIG_32.ini/",
             "./NORMAL_CONFIG_33.ini/",
             "./NORMAL_CONFIG_41.ini/",
             "./NORMAL_CONFIG_42.ini/",
             "./NORMAL_CONFIG_43.ini/"]

import os

for dir_name in dir_names:
    for root, dirs, files in os.walk(dir_name):
        for d in dirs:
            if d[:7] == "output_":
                path = os.path.join(root, d)
                old_fn = path + "/MDF_PER.dat"
                new_fn = path + "/MDF_PER_CLEAN.dat"
                os.system("sed '1,13d' {} > {}".format(old_fn, new_fn))
                os.system("rm {}".format(old_fn))
