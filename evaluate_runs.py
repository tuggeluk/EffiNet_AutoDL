import os
import pandas as pd
import numpy as np

df = pd.DataFrame(columns=['AUC', 'Dataset', 'LearningRate', 'LRPatience', 'LRDecay', 'WeightDecay', 'MaximumEdge',
                           'FullyConnected', 'Network', 'Pretraining', 'UseSoftmaxXent', 'BatchSize', 'DropoutRate'])

runs = os.listdir(".")
runs = [x for x in runs if x[0:7] == "result_"]

ind = 0
for i_run, run in enumerate(runs):
    run_files = os.listdir(run)
    run_files = [x for x in run_files if ".txt" in x]

    configs = run.split("_")

    for i_file, run_file in  enumerate(run_files):
        with open(os.path.join(run,run_file), encoding="utf-8") as search:
            for line in search:
                if "Final area under learning curve" in line:
                    line = line.strip()
                    line_split = line.split(" ")
                    perfomance = float(line_split[-1])
                    data = line_split[-2][:-1]

                    df.loc[ind] = [perfomance,data,float(configs[1]),int(configs[2]),float(configs[3]),float(configs[4]),int(configs[5]),
                                   int(configs[6]), configs[7], configs[8], configs[9], int(configs[10]), float(configs[11])]
                    ind +=1
                    break


pd.set_option('display.expand_frame_repr', False)

print("*************************************")
print("***** Top perfomers per Dataset *****")
print("*************************************")
for ds in df.Dataset.unique():
    print(df[df.Dataset == ds].sort_values(by='AUC', ascending=False)[0:3])

globalAUC = df.AUC.mean().round(5)
for conf_of_interest in ["LearningRate", 'WeightDecay', 'MaximumEdge', 'Network', 'UseSoftmaxXent', 'BatchSize', 'DropoutRate']:
    print("****************************************")
    print("***** Impact of individual "+conf_of_interest+" *****")
    print("****************************************")

    df_lr = pd.DataFrame(columns=['AUC', conf_of_interest, 'DiffAUC', 'NrRuns'])
    for ix, lr in enumerate(df[conf_of_interest].unique()):
        aucs = df[df[conf_of_interest] == lr].AUC
        amean = aucs.mean().round(5)
        df_lr.loc[ix] = [amean,lr, (amean-globalAUC), aucs.__len__()]
    print(df_lr.sort_values(by='AUC', ascending=False))





