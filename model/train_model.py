#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path

from build_model import CNNmodel
from utils import mse_result, pearson_result, save_weights

mode = "merge"   # options: "single_dnam", "single_rna", "merge"
rounds = 10

dir_input = "/home/CBBI/tsaih/data/DNAm/"

if mode == "single_rna":
    dir_output = "/home/CBBI/tsaih/Research_DNAm/diffX/"
    tmpName = "DeepG2Pstructure_RNA"

elif mode == "single_dnam":
    dir_output = "/home/CBBI/tsaih/Research_DNAm/diffX/"
    tmpName = "DeepG2Pstructure_DNAm"

elif mode == "merge":
    dir_output = "/home/CBBI/tsaih/Research_DNAm/EarlymergeX_manual/"
    tmpName = "DeepG2Pstructure_EarlymergeManual"

Path(dir_output).mkdir(parents=True, exist_ok=True)


# Load data

pro1 = pd.read_csv(f"{dir_input}/Protein_train.txt", sep="\t", index_col=0).T
pro2 = pd.read_csv(f"{dir_input}/Protein_val.txt", sep="\t", index_col=0).T
pro3 = pd.read_csv(f"{dir_input}/Protein_test.txt", sep="\t", index_col=0).T

if mode == "single_dnam":
    x1 = pd.read_csv(f"{dir_input}/DNAm_train.txt", sep="\t", index_col=0).T
    x2 = pd.read_csv(f"{dir_input}/DNAm_val.txt", sep="\t", index_col=0).T
    x3 = pd.read_csv(f"{dir_input}/DNAm_test.txt", sep="\t", index_col=0).T

elif mode == "single_rna":
    x1 = pd.read_csv(f"{dir_input}/RNA_train.txt", sep="\t", index_col=0).T
    x2 = pd.read_csv(f"{dir_input}/RNA_val.txt", sep="\t", index_col=0).T
    x3 = pd.read_csv(f"{dir_input}/RNA_test.txt", sep="\t", index_col=0).T

elif mode == "merge":
    meth1 = pd.read_csv(f"{dir_input}/DNAm_train.txt", sep="\t", index_col=0).T
    meth2 = pd.read_csv(f"{dir_input}/DNAm_val.txt", sep="\t", index_col=0).T
    meth3 = pd.read_csv(f"{dir_input}/DNAm_test.txt", sep="\t", index_col=0).T
    exp1 = pd.read_csv(f"{dir_input}/RNA_train.txt", sep="\t", index_col=0).T
    exp2 = pd.read_csv(f"{dir_input}/RNA_val.txt", sep="\t", index_col=0).T
    exp3 = pd.read_csv(f"{dir_input}/RNA_test.txt", sep="\t", index_col=0).T


# Training start

tableCNN = pd.DataFrame(columns=[
    "mse_train", "mse_val", "mse_test",
    "cor_train", "cor_val", "cor_test", "epochs"
])

for i in range(rounds):
    print(f"{mode}: Round {i+1}")

    if mode == "merge":
        resultCNN = CNNmodel(exp1, exp2, exp3,
                             pro1, pro2, pro3,
                             x_train2=meth1, x_val2=meth2, x_test2=meth3)
    else:  # single_dnam or single_rna
        resultCNN = CNNmodel(x1, x2, x3,
                             pro1, pro2, pro3)

    pearsonCNN = pearson_result(pro1, pro2, pro3,
                               resultCNN[3], resultCNN[4], resultCNN[5])

    # Save metrics
    tableCNN.loc[i, "mse_train"] = resultCNN[0]
    tableCNN.loc[i, "mse_val"]   = resultCNN[1]
    tableCNN.loc[i, "mse_test"]  = resultCNN[2]
    tableCNN.loc[i, "cor_train"] = pearsonCNN["pearsonr_train"].mean()
    tableCNN.loc[i, "cor_val"]   = pearsonCNN["pearsonr_val"].mean()
    tableCNN.loc[i, "cor_test"]  = pearsonCNN["pearsonr_test"].mean()
    tableCNN.loc[i, "epochs"]    = resultCNN[6]

    # Save detailed results
    pearsonCNN.to_csv(f"{dir_output}{tmpName}_corr_results_{i+1}.txt", sep="\t", index=False)
    resultCNN[3].to_csv(f"{dir_output}{tmpName}_predfromtrain_{i+1}.txt", sep="\t")
    resultCNN[4].to_csv(f"{dir_output}{tmpName}_predfromval_{i+1}.txt", sep="\t")
    resultCNN[5].to_csv(f"{dir_output}{tmpName}_predfromtest_{i+1}.txt", sep="\t")
    resultCNN[7].save(f"{dir_output}{tmpName}_model_{i+1}.h5")
    save_weights(resultCNN[7], f"{dir_output}{tmpName}_model_{i+1}.pickle")

tableCNN.to_csv(f"{dir_output}Table_result_{rounds}times_{tmpName}.txt", sep="\t")


