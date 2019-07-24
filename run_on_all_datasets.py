import os




l_rate = [0.0005,0.005]
w_decay = [0.00002]
m_edge = [256]
fc_size = [256]
datasets = ["Hammer/", "Pedro/", "Decal/", "Chucky/", "cifar100/"]
#datasets = ["Hammer/"]

time_budget = 1200
code_dir = "./"
base_dataset_dir = "autodl/AutoDL_public_data/"

for fc in fc_size:
    for me in m_edge:
        for lr in l_rate:
            for wd in w_decay:
                for ds in datasets:

                    folder = "./eff_result_"+str(lr)+"_"+str(wd)+"_"+str(me)+"_"+str(fc)
                    try:
                        os.mkdir(folder)
                    except:
                        pass

                    copy_command = "cp autodl/AutoDL_scoring_output/learning-curve-*.png "+folder

                    text_file = open("config.txt", "w")
                    text_file.write(str(lr)+"_"+str(wd)+"_"+str(me)+"_"+str(fc))
                    text_file.close()

                    command_execution = "python {} --dataset_dir={} --code_dir={} --time_budget={}".format("autodl/run_local_test.py", base_dataset_dir+ds, code_dir, time_budget)
                    command_execution = command_execution + " > " + folder + "/console"+ds[0:3]+".txt"
                    os.system(command_execution)
                    os.system(copy_command)