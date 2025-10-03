#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from constant_variables import *
from glob import glob
import re
from tqdm import tqdm


# In[3]:


tumorType='LUSC'
pro = 'P53'
t = 'high'

dir_input='/home/CBBI/tsaih/Research_DNAm/EarlymergeX_manual/IG/GradientDistribution_HighvsLow_proabd_base0/' + tumorType+ '/'
dir_output=dir_input+"Plot_IGDistribution/"
Path(dir_output).mkdir(parents=True, exist_ok=True)

df_perc='/home/CBBI/tsaih/Research_DNAm/EarlymergeX_manual/IG/GradientDistribution_HighvsLow_proabd_base0/GradientDistribution_Analysis/'+        'Table_numFeaturesperProtein_exp'+ 'high' + '_' + 'pos' + '.txt'
df_perc=pd.read_csv(df_perc, sep='\t')
df_perc.Protein=df_perc.Protein.str.upper()

probes=pd.read_csv("/home/CBBI/tsaih/data/"+"Table_ProbeInformation_organized.txt", sep='\t')

Pairs=CorrRNAnPro()
Pairs.proName=Pairs.proName.str.upper()
Pairs.geneName=Pairs.geneName.str.upper()

# Read gradient file
f = dir_input + 'gradient_mean_HvsL_' + pro + '.txt'
pairGene = Pairs[Pairs.proName == pro]['geneName'].tolist()[0]

df=pd.read_csv(f , sep='\t', index_col=[0])
df=pd.merge(df, probes[['Name', 'UCSC_RefGene_Name', 'UCSC_RefGene_Group']], left_index=True, right_on='Name', how='left')
df.index=df.Name
df=df.sort_values(by=t+'_rank')
df['omic']=df['omic'].str.replace('methyl', 'Methyl')


# In[4]:


#----------Figure 3A----------------

fig, ax = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

# Plot the second subplot (bar plot)
df_plot = df_perc[(df_perc.Protein == pro) & (df_perc['tumorType'] == tumorType)]
ax[1].barh(pro, 100, color='orange')
ax[1].barh(pro, df_plot['Perc_methyl'], color='green')
ax[1].set_ylabel('% Methyl contribution')
ax[1].set_title('%Methyl: ' + str(df_plot['Perc_methyl'].tolist()))

# Plot the first subplot (density plot)
sns.distplot(df[t + '_zscore'], kde=False, ax=ax[0])
sns.rugplot(data=df, x=t + '_zscore', hue='omic',hue_order=['Methyl', 'RNA'], palette=['green', 'orange'], ax=ax[0])

ax[0].axvline(x=df.loc[pairGene][t + '_zscore'].tolist(), color='orange')
ax[0].set_title("Protein:" + pro, fontsize=22)
ax[0].set_xlabel("Integrated gradient (zscore)", fontsize=18)
ax[0].set_ylabel("Number of genes", fontsize=18)

# Additional ticks at -1.96 and 1.96
ax[0].set_xticks(list(ax[0].get_xticks()) + [-1.96, 1.96])
ax[0].set_yticklabels(ax[0].get_yticks().astype(int), size=14)
ax[0].text(df.loc[pairGene][t + '_zscore'].tolist() + 0.0001, 200 + 50,
                 "{}: {}".format(df.loc[pairGene][t + '_rank'].astype(int),
                                 pairGene), fontsize=17)

# Add text for the first methyl probe
index_of_first_M = df['omic'].str.find('Methyl').idxmax()
ax[0].axvline(x=df.loc[index_of_first_M][t + '_zscore'].tolist(), color='green')
ax[0].text(df.loc[index_of_first_M][t + '_zscore'].tolist() + 0.0001, 200 + 50,
                   "{}:{}({}),{}".format(df.loc[index_of_first_M][t + '_rank'],
                                         df.loc[index_of_first_M]['UCSC_RefGene_Name'],
                                         df.loc[index_of_first_M]['UCSC_RefGene_Group'],
                                         index_of_first_M,),
                   fontsize=12)

# Add text for the first promoter methyl probe
index_of_first_M = df[df.omic=='Methyl']
index_of_first_M=index_of_first_M.reset_index(drop=True)
condition_satisfied = False
for index, row in index_of_first_M.iterrows():
    if condition_satisfied:
        break
    if pd.notna(row['UCSC_RefGene_Group']):
        groups = row['UCSC_RefGene_Group'].split(';') # Split the 'Group' string by ';'
        for g in range(len(groups)):
            group=groups[g]
            if group in ['TSS1500', 'TSS200', '1stExon', "5'UTR"]:
                name_first_P=row['UCSC_RefGene_Name'].split(';')[g]
                group_first_P=group
                probe_first_p=row['Name']
                condition_satisfied=True
                break  # Exit the loop after finding the first row

ax[0].axvline(x=df[df.Name==probe_first_p][t + '_zscore'][0], color='green')
ax[0].text(df[df.Name==probe_first_p][t + '_zscore'][0] + 0.0001, 200 + 50,
                   "{}:{}({}),{}".format(df[df.Name==probe_first_p][t + '_rank'][0],
                                           name_first_P,
                                           group_first_P,
                                           probe_first_p,),
                   fontsize=12)

# Set ylabel for the first subplot
ax[0].set_ylabel('Number of features')

# Adjust layout
plt.tight_layout()

# plt.savefig(dir_output + "Densityplot_" + "IG_" + pro + '.png', format='png', dpi=600, transparent=True)
# plt.savefig(dir_output + "Densityplot_" + "IG_" + pro + '.pdf', format='pdf', dpi=600, transparent=True)
# plt.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




