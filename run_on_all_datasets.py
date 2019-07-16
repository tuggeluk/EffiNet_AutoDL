import os



datasets = ["Hammer/", "Pedro/", "Decal/", "Chucky/"]
time_budget = 1200
code_dir = "./"
base_dataset_dir = "autodl/AutoDL_public_data/"

copy_command = "cp autodl/AutoDL_scoring_output/learning-curve-*.png ./result"

for ds in datasets:
    command_execution = "python {} --dataset_dir={} --code_dir={} --time_budget={}".format("autodl/run_local_test.py", base_dataset_dir+ds, code_dir, time_budget)
    os.system(command_execution)
    os.system(copy_command)