@author: bhushan

The data needed to run the `synthetic_data_experiment_outputs.ipynb` lives in Dropbox in Experimental > normal/synthetic_experiment. 
The input used for the runs also lives there.

* `clean_person_file.py` is a quick and dirty script to remove the first 13 lines from each Person file, which is just metadata preamble from the DAS and prevents the file from being read as a Pandas Dataframe.
* `del_extras.py` is a script to delete every thing in the runs but th MDF_PER.dat files, which we will use. I used this because otherwise the directory would bloat up astronomically and my Mac would yell at me.
* `build_input_file_for_normal_experiment.py` is the file I used to generate the input file for this experiment. It requires a alabama_1940s_raw.dat file as an input which is just the 1940s Alabama lines. TODO: add this up to Dropbox.

