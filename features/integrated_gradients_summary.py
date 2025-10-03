#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from glob import glob
import re
from tqdm import tqdm
from constant_variables import *
from scipy import stats
from itertools import combinations
from pathlib import Path

dir_data='/home/CBBI/tsaih/data/DNAm/'
exp_train=pd.read_csv(dir_data + "RNA_train.txt", sep='\t', index_col=[0])
exp_val=pd.read_csv(dir_data + "RNA_val.txt", sep='\t', index_col=[0])
exp_test=pd.read_csv(dir_data + "RNA_test.txt", sep='\t', index_col=[0])
exp=pd.merge(pd.merge(exp_train, exp_val, left_index=True, right_index=True), exp_test, left_index=True, right_index=True)
del exp_train, exp_val, exp_test

met_train=pd.read_csv(dir_data + "DNAm_train.txt", sep='\t', index_col=[0])
met_val=pd.read_csv(dir_data + "DNAm_val.txt", sep='\t', index_col=[0])
met_test=pd.read_csv(dir_data + "DNAm_test.txt", sep='\t', index_col=[0])
met=pd.merge(pd.merge(met_train, met_val, left_index=True, right_index=True), met_test, left_index=True, right_index=True)
del met_train, met_val, met_test

Ytrain=pd.read_csv(dir_data + "Protein_train.txt", sep='\t', index_col=[0]).T
Yval=pd.read_csv(dir_data + "Protein_val.txt", sep='\t', index_col=[0]).T
Ytest=pd.read_csv(dir_data + "Protein_test.txt", sep='\t', index_col=[0]).T
Y=pd.concat([Ytrain, Yval, Ytest])
del Ytrain, Yval, Ytest

Y.columns=Y.columns.str.replace('_p', '_P')

Pairs=CorrRNAnPro()
Pairs.proName=Pairs.proName.str.upper()
Pairs.geneName=Pairs.geneName.str.upper()

modeltype='merge'
datatype='DNAm'
IGprocess='base0'
tumortype='BRCA'

if modeltype=='merge':
    dir_input = '/home/CBBI/tsaih/Research_DNAm/EarlymergeX_manual/IG/'
else:
    dir_input='/home/CBBI/tsaih/Research_DNAm/diffX/IG_' +modeltype +'/'

if modeltype=='merge':
    data=pd.concat([exp, met])
elif modeltype=='RNA':
    data=exp
elif modeltype=='DNAm':
    data=met

data=data.T

PromoterCor=pd.read_csv('/home/CBBI/tsaih/Research_DNAm/Correlation_DNAmethRNA/'+
                        'Table_correlation_protein2RNA2DNAmeth_Promoters' + '.txt', sep='\t')
PromoterCor.proName=PromoterCor.proName.str.upper()

#tumor types
# Set default values for tumortype and IGprocess
samples=samples2tumortypes()
samples=samples[samples['sample'].isin(data.index)].reset_index(drop=True)
#targetTumors=samples.Disease.unique()
#targetTumors=np.append(targetTumors, 'PanCan')
targetTumors=['PanCan']

# if tumortype == '':
#     tumortype = 'PanCan'

# Construct dir_output
dir_output = dir_input + "GradientDistribution_HighvsLow_proabd_" + IGprocess + "/" + tumortype + "/"
#dir_output = dir_input + "GradientDistribution_HighvsLow_proabd_" + IGprocess + "/"
Path(dir_output).mkdir(parents=True, exist_ok=True)

filenames = glob(dir_input + 'gradient_' + '*.txt')

# done_files= glob(dir_output + 'gradient_mean_HvsL_' + '*.txt')
# done_pro=[]

# for d in done_files:
#     done_pro.append(re.search('gradient_mean_HvsL_' + '(.*).txt', d).group(1))

grad_summarys=[]
#f=dir_input+'gradient_P53.txt'
#pro='ERALPHA'
for i in tqdm(range(len(filenames))):
    #print(f)
    f=filenames[i]
    pro=re.search('gradient_' + '(.*).txt', f).group(1)
    print("{}: {}".format(list.index(filenames, f)+1, pro))
    # if pro in done_pro:
    #     print ('    done')
    #     df_result = pd.read_csv(dir_output+'gradient_mean_HvsL_' + pro+'.txt', sep="\t", index_col=[0])
    #
    # else:
    gradOri=pd.read_csv(f, sep="\t", header=None).T
    gradOri.index=Y.index
    gradOri.columns=data.columns

    if IGprocess=='base0+revertInput':
        gradOri=gradOri/data

    Y_target=pd.DataFrame(Y[pro])

    for targetTumor in targetTumors:
        dir_output = dir_input + "GradientDistribution_HighvsLow_proabd_" + IGprocess + "/" + targetTumor + "/"
        Path(dir_output).mkdir(parents=True, exist_ok=True)

        # select tumor samples
        if targetTumor=='PanCan':
            samples_target = samples['sample'].tolist()
        else:
            samples_target = samples.loc[samples['Disease'] == targetTumor, 'sample'].tolist()
        nTumor = len(samples_target)
        gradOri_tumor = gradOri.loc[samples_target, :]
        Y_target_df = Y_target.loc[samples_target, :]

        df_result=[]
        for t in [ 'full', 'high', 'low']:
            if t=='high':
                targetSamples=list(Y_target_df[pro].nlargest(n=round(nTumor*0.1)).index)
            elif t=='low':
                targetSamples = list(Y_target_df[pro].nsmallest(n=round(nTumor* 0.1)).index)
            elif t=='full':
                targetSamples = list(Y_target_df[pro].nsmallest(n=round(nTumor)).index)

            df=pd.DataFrame(gradOri.loc[targetSamples,:].mean(), columns=[t+'_ori'])
            df[t+'_zscore'] = stats.zscore(df[t+'_ori'])
            df[t+'_rank']=df[t+'_ori'].rank(ascending=False)
            df_result.append(df)
        df_result=pd.concat(df_result, axis=1)
        df_result=df_result.sort_values(by='high_rank', ascending=True)
        df_result['omic']=np.where(df_result.index.isin(met.index), 'Methyl', 'RNA')

        df_result.to_csv(dir_output + 'gradient_mean_HvsL_' + pro + '.txt', sep='\t', header=True, index=True)

        #if datatype!='DNAm':
        # Initialize lists
        metrics = ['full', 'high', 'low']
        summary_data = {f'{metric}_{stat}': [] for metric in metrics for stat in ['rank', 'mean', 'zscore', 'max', 'min']}
        proName, entityName, entityType = [], [], []

        # Get corresponding gene and methylation names
        pair_gene = Pairs.loc[Pairs['proName'] == pro, 'geneName'].values[0]
        pair_meth = PromoterCor.loc[PromoterCor['proName'] == pro, '#id'].values[0]


        # Function to process and append results for either gene or meth
        def append_results(pair, entity):
            proName.append(pro)
            entityName.append(pair)  # Store either gene or meth in entityName
            entityType.append(entity)  # Store whether it's 'gene' or 'meth'

            for metric in metrics:
                if pair == '.' or pair not in df_result.index.to_list():
                    # Append '.' when pair is '.'
                    summary_data[f'{metric}_rank'].append('.')
                    summary_data[f'{metric}_mean'].append('.')
                    summary_data[f'{metric}_zscore'].append('.')
                    summary_data[f'{metric}_max'].append('.')
                    summary_data[f'{metric}_min'].append('.')
                else:
                    # Append valid data for the given pair
                    summary_data[f'{metric}_rank'].append(df_result.loc[pair, f'{metric}_rank'].tolist())
                    summary_data[f'{metric}_mean'].append(df_result.loc[pair, f'{metric}_ori'].tolist())
                    summary_data[f'{metric}_zscore'].append(df_result.loc[pair, f'{metric}_zscore'].tolist())
                    summary_data[f'{metric}_max'].append(df_result[f'{metric}_ori'].max())
                    summary_data[f'{metric}_min'].append(df_result[f'{metric}_ori'].min())


        # Append results for pair_gene
        append_results(pair_gene, 'gene')
        # Append results for pair_meth
        append_results(pair_meth, 'meth')

        # Create summary DataFrame with the new combined column 'entityName' and 'entityType'
        grad_summary = pd.DataFrame({
            'proName': proName,
            'entityName': entityName,  # Combined column for gene and meth
            'entityType': entityType,  # Column to indicate gene or meth
            'tumorType': targetTumor,
            **summary_data
        })
        grad_summarys.append(grad_summary)
grad_summarys=pd.concat(grad_summarys)
grad_summarys.to_csv(dir_output + 'gradient_mean_summary2'+ '.txt', sep='\t', header=True, index=True)
print('Finihsed')



