import os




l_rate = [0.001]
l_rate_patience = [100]
l_rate_decay = [0.5]
w_decay = [0.000025]
m_edge = [128]
fc_size = [256]
datasets = ["Hammer/", "Pedro/", "Decal/", "cifar100/"]
#datasets = ["Hammer/", "Decal/"]
#datasets = ["Decal/"]
network_code = ["efficientnet-bmini"]
pretraining = ["imagenet"]
bottleneck_fc = [False]
batch_size = [32]
dropout = [0.5]


time_budget = 1200
code_dir = "./"
base_dataset_dir = "autodl/AutoDL_public_data/"

for dr in dropout:
    for bs in batch_size:
        for bn in bottleneck_fc:
            for pt in pretraining:
                for nwc in network_code:
                    for ds in datasets:
                        for fc in fc_size:
                            for me in m_edge:
                                for wd in w_decay:
                                    for lrd in l_rate_decay:
                                        for lrp in l_rate_patience:
                                            for lr in l_rate:

                                                folder = "./result"+"_"+str(lr)+"_"+str(lrp)+"_"+str(lrd)\
                                                         +"_"+str(wd)+"_"+str(me)+"_"+str(fc) \
                                                         + "_" + str(nwc)+ "_" + str(pt)+ "_" + str(bn)+"_" + str(bs)+"_" + str(dr)
                                                try:
                                                    os.mkdir(folder)
                                                except:
                                                    pass

                                                copy_command = "cp autodl/AutoDL_scoring_output/learning-curve-*.png "+folder

                                                text_file = open("config.txt", "w")
                                                text_file.write(str(lr)+"_"+str(lrp)+"_"+str(lrd)
                                                         +"_"+str(wd)+"_"+str(me)+"_"+str(fc)
                                                         + "_" + str(nwc)+ "_" + str(pt)+ "_" + str(bn) + "_" + str(bs) + "_" + str(dr))
                                                text_file.close()

                                                command_execution = "python {} --dataset_dir={} --code_dir={} --time_budget={}".format("autodl/run_local_test.py", base_dataset_dir+ds, code_dir, time_budget)
                                                command_execution = command_execution + " > " + folder + "/console"+ds[0:3]+".txt"
                                                os.system(command_execution)
                                                os.system(copy_command)