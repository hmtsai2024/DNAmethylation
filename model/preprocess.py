#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from constant_variables import *

dir_input = "/home/CBBI/tsaih/data/"
dir_output=dir_input+'DNAm/'

from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

#------------------------
# Training data
#------------------------
sample_train=sampleTrain()
sample_val=sampleVal()
sample_test=sampleTest()

methy='jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv'
methy=pd.read_csv(dir_input+methy, sep='\t', index_col=[0])

methy=methy.fillna(0)

#filter probes
#We eliminated probes with low methylation(beta value, <0.3) in >90% of TCGA samples,
# Step 1: Identify probes with values less than 0.3 in each sample
mask = methy < 0.3
# Step 2: Calculate the percentage of samples where the values are less than 0.3 for each probe
probe_percentages = mask.mean(axis=1) * 100
# Step 3: Filter out probes that have less than 0.3 values in more than 90% of samples
filtered_probes = probe_percentages[probe_percentages <= 90]

#filter samples from 6813
sample_methy=pd.DataFrame(methy.columns, columns=['ID'])
sample_methy['IDshort']=sample_methy.ID.str.slice(stop=15)
sample_methy['IDshort']=sample_methy['IDshort'].str.replace('-','.')
sample_methy=sample_methy[sample_methy.IDshort.isin(sample_train+ sample_val+sample_test)] #6750
sample_methy=sample_methy.drop_duplicates(subset='IDshort', keep='first')

methy3=methy.loc[filtered_probes.index, sample_methy['ID']]
methy3.columns=methy3.columns.str.slice(stop=15)
methy3.columns=methy3.columns.str.replace('-', '.')

def intersect(A, B):
    C=list(set(A)&set(B))
    return C

sample_train2=sorted(intersect(sample_train, methy3.columns))
sample_val2=sorted(intersect(sample_val, methy3.columns))
sample_test2=sorted(intersect(sample_test, methy3.columns))

methy3_train=methy3[sample_train2]
methy3_val=methy3[sample_val2]
methy3_test=methy3[sample_test2]

methy3_train.to_csv(dir_output + 'DNAm_train.txt', sep='\t')
methy3_val.to_csv(dir_output + 'DNAm_val.txt', sep='\t')
methy3_test.to_csv(dir_output + 'DNAm_test.txt', sep='\t')

# Prepare gene expression data
data=['train', 'val', 'test']

data=pd.read_csv(dir_input + 'X_data_batch_13995x6813_train_z.txt', sep="\t", index_col=[0])
data=data[sample_train2]
data.to_csv(dir_output + 'RNA_train.txt', sep='\t')

data=pd.read_csv(dir_input + 'X_data_batch_13995x6813_val_z.txt', sep="\t", index_col=[0])
data=data[sample_val2]
data.to_csv(dir_output + 'RNA_val.txt', sep='\t')

data=pd.read_csv(dir_input + 'X_data_batch_13995x6813_test_z.txt', sep="\t", index_col=[0])
data=data[sample_test2]
data.to_csv(dir_output + 'RNA_test.txt', sep='\t')

# Prepare protein abundance data
data=pd.read_csv(dir_input + 'Y_data_187x6813_train.txt', sep="\t", index_col=[0])
data=data[sample_train2]
data.to_csv(dir_output + 'Protein_train.txt', sep='\t')

data=pd.read_csv(dir_input + 'Y_data_187x6813_val.txt', sep="\t", index_col=[0])
data=data[sample_val2]
data.to_csv(dir_output + 'Protein_val.txt', sep='\t')

data=pd.read_csv(dir_input + 'Y_data_187x6813_test.txt', sep="\t", index_col=[0])
data=data[sample_test2]
data.to_csv(dir_output + 'Protein_test.txt', sep='\t')

#------------------------
# Probe data
#------------------------
dir_input = "/home/CBBI/tsaih/data/DNAm/"
probes=pd.read_csv(dir_input + 'DNAm_test' + '.txt', sep="\t", index_col=[0])
probes=probes.index.tolist()

map27=pd.read_csv('/home/CBBI/tsaih/data/' + 'probeMap_illuminaMethyl27K_hg18_gpl8490_TCGAlegacy', sep="\t")
map45=pd.read_csv('/home/CBBI/tsaih/data/' + 'probeMap_illuminaMethyl450_hg19_GPL16304_TCGAlegacy', sep="\t")
map27['data']='27K'
map45['data']='450K'

probeMap=pd.concat([map45, map27])
probeMap['gene']=probeMap['gene'].fillna('.')
probeMap=probeMap.sort_values(by=['gene', 'chrom'])
probeMap=probeMap.reset_index(drop=True)
# Drop duplicates based on 'id', keeping the first occurrence (which is the 450k data)
probeMap = probeMap.sort_values(by='data', ascending=False)
probeMap = probeMap.drop_duplicates(subset='#id', keep='first')
probeMap_target=probeMap[probeMap['#id'].isin(probes)]
probeMap_target=probeMap_target.sort_values(by=['gene', 'chrom'])
probeMap_target=probeMap_target.reset_index(drop=True)

#promoters
PromoterProbe=pd.read_csv("/home/CBBI/tsaih/data/DNAm/"+'PromoterProbesfromPaper2023CancerCell.txt', sep='\t')
probeMap_target=pd.merge(probeMap_target, PromoterProbe, left_on='#id', right_on='ProbeIndex', how='left')
probeMap_target.rename(columns={'ProbeIndex': 'PromoterProbes'}, inplace=True)

probeMap_target.to_csv(dir_input+'Table_DNAmprobe10199_Information.txt', sep='\t', index=False)


probes_promoter=pd.read_csv("/home/CBBI/tsaih/data/DNAm/"+'Table_DNAmprobe10199_Information.txt', sep='\t')
probes_promoter=probes_promoter[['PromoterProbes', 'Gene']].drop_duplicates().dropna().reset_index(drop=True)
#aggregat genenames if probe id is the same
probes_promoter=probes_promoter.groupby('PromoterProbes')['Gene'].agg(lambda x: ';'.join(map(str, x))).reset_index()

probes_all=pd.read_csv("/home/CBBI/tsaih/data/DNAm/"+'ProbeInformation.txt', sep='\t')
probes_all=probes_all[['Name', 'chr', 'Relation_to_Island', 'UCSC_RefGene_Accession', 'UCSC_RefGene_Name', 'UCSC_RefGene_Group',
                            'DMR', 'Enhancer', 'Regulatory_Feature_Group', 'DHS']].drop_duplicates().reset_index(drop=True)
probes=pd.merge(probes_promoter, probes_all, left_on='PromoterProbes', right_on='Name', how='outer')
probes['Promoters']=np.where(probes['PromoterProbes'].notna(), 'promoter', '.')
col_promotergene=probes['Gene']
probes=probes.drop(['PromoterProbes', 'Gene'], axis=1)
probes['Gene']=col_promotergene
del probes_promoter, probes_all

probes2=probes.copy()

df_new = []
for index, row in probes2.iterrows():
    if pd.isna(row['UCSC_RefGene_Group']):
        df_new.append(row)  # Append original row if either column is NaN
    else:
        accessions = row['UCSC_RefGene_Accession'].split(';')
        groups = row['UCSC_RefGene_Group'].split(';')
        names = row['UCSC_RefGene_Name'].split(';')
        for accession, group, name in zip(accessions, groups, names):
            #print(group+'_'+name)
            new_row = row.copy()
            new_row['UCSC_RefGene_Accession_new'] = accession.strip()
            new_row['UCSC_RefGene_Group_new'] = group.strip()
            new_row['UCSC_RefGene_Name_new'] = name.strip()
            df_new.append(new_row)

# Concatenate the exploded DataFrames along the axis
result_df = pd.concat(df_new, axis=1).T.reset_index(drop=True) #806789
result_df=result_df.drop_duplicates().reset_index(drop=True) #806731
result_df['UCSC_RefGene_Group_new_promoter']=np.where(result_df['UCSC_RefGene_Group_new'].isin(['TSS1500', 'TSS200', '1stExon', "5'UTR"]),
                                           'promoters', result_df['UCSC_RefGene_Group_new'])
result_df=result_df.drop_duplicates(
    subset=['Name', 'UCSC_RefGene_Accession_new', 'UCSC_RefGene_Name_new', 'UCSC_RefGene_Group_new_promoter']).reset_index(drop=True) #769032

result_df.to_csv("/home/CBBI/tsaih/data/"+"Table_ProbeInformation_organized.txt", sep='\t', index=False)


