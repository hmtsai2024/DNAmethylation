#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns
from scipy.stats import ttest_ind
from itertools import combinations
from constant_variables import *
from scipy.stats import pearsonr, spearmanr

dir_input='/home/CBBI/tsaih/Research_DNAm/'
dir_output='/home/CBBI/tsaih/Research_DNAm/Bestmodel_resultAnalysis/'
from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

df_RNA=pd.read_csv(dir_input + '/diffX/DeepG2Pstructure_RNA_corr_results_2.txt', sep="\t", index_col=[0])
df_DNAm=pd.read_csv(dir_input + '/diffX/DeepG2Pstructure_DNAm_corr_results_6.txt', sep="\t", index_col=[0])
df_merge=pd.read_csv(dir_input+'/EarlymergeX_manual/DeepG2Pstructure_EarlymergeManual_corr_results_8.txt', sep='\t', index_col=[0])

df_RNA.columns=df_RNA.columns+'_RNA'
df_DNAm.columns=df_DNAm.columns+'_DNAm'
df_merge.columns=df_merge.columns+'_merge'
combined_df=pd.concat([df_DNAm[['pearsonr_test_DNAm', 'pearsonp_test_DNAm']],
                       df_RNA[['pearsonr_test_RNA', 'pearsonp_test_RNA']],
                       df_merge[['pearsonr_test_merge','pearsonp_test_merge']]], axis=1)
combined_df['diff']=combined_df['pearsonr_test_merge']-combined_df['pearsonr_test_RNA']
combined_df['protein']=np.where(combined_df.index.str.contains('_p'), 'Phosph', 'Total')
combined_df['protein']=np.where(combined_df.index.str.contains('ACET'), 'Acetyl', combined_df['protein'])
combined_df=combined_df.sort_values(by='diff', ascending=False)

#------Figure 2A------

combined_df['baseRNA_merge']=combined_df['pearsonr_test_merge']-combined_df['pearsonr_test_RNA']
combined_df=combined_df.sort_values(by='baseRNA_merge', ascending=False)

sns.barplot(data=combined_df, x=combined_df.index, y='baseRNA_merge', color='blue', label='RNA+Methyl')

plt.title('Performance of three models using RNA as baseline')
plt.xlabel('Proteins')
plt.ylabel('Pearson correlation')
plt.legend(title='Models', loc='upper right')
plt.tight_layout()
#plt.savefig(dir_output + "Barplot_" + "performanceofthreemodels_perprotein_baseRNA" + '.png', format='png', dpi=600, transparent=True)
#plt.savefig(dir_output + "Barplot_" + "performanceofthreemodels_perprotein_baseRNA" + '.pdf', format='pdf', dpi=600, transparent=True)
#plt.close()

#------Figure 1C------
#plot pie best
combined_df['Best_Model'] = combined_df[['pearsonr_test_RNA', 'pearsonr_test_DNAm', 'pearsonr_test_merge']].idxmax(axis=1)
combined_df['Best_Model']=combined_df['Best_Model'].str.replace('pearsonr_test_', '')
combined_df=combined_df.loc[combined_df['protein'] !='Acetyl']
combined_df['diff_word']=np.where(combined_df['diff']>0, 'MethImproved', 'Non-MethImproved')

# Count the number of genes that perform best in each model
model_counts = combined_df['Best_Model'].value_counts()
order=['merge', 'RNA', 'DNAm']
model_counts=model_counts[order]
print(model_counts)

# Plot a pie chart
plt.pie(model_counts, labels=model_counts.index, autopct='%1.1f%%', startangle=165)
plt.title('Percentage of Proteins Performing Best in Each Model')
plt.tight_layout()
# plt.savefig(dir_output + "Piechart_" + "bestmodelbestinput" + '.png', format='png', dpi=600, transparent=True)
# plt.savefig(dir_output + "Piechart_" + "bestmodelbestinput" + '.pdf', format='pdf', dpi=600, transparent=True)
# plt.close()

#------Figure 2B------

dir_data='/home/CBBI/tsaih/data/'
ProtGrp=pd.read_csv(dir_data+'ProteinGroupsSummary.txt', sep='\t')
ProtGrp['rppa']=ProtGrp['rppa'].str.replace('_P', '_p')
df=pd.merge(combined_df, ProtGrp[['rppa', 'Corum_sublocation)', 'Panther_class', 'Panther_class2']], left_index=True, right_on='rppa')
df=df[~df.duplicated()]

df['Panther_class']=np.where(df['Panther_class']=='0', 'Others', df['Panther_class'])
df['Panther_class']=df['Panther_class'].str.capitalize()

df_count=pd.DataFrame(df.groupby(['Panther_class', 'protein']).count()['rppa']).reset_index().sort_values(by='rppa')

class2others=[]
for p in df_count['Panther_class']:
    #print(p)
    a=df_count[df_count['Panther_class']==p]
    #check if both Total and Phospho exist
    if all(x in a['protein'] for x in ['Total', 'Phosph']):
        if a[a['protein']=='Total']['rppa'].values<2 and a[a['protein']=='Phosph']['rppa'].values<2:
            class2others.append(p)
    else:
        mod=a['protein'].tolist()
        if ('Acetyl' not in mod) and (len(mod)<2):
            if a[a['protein']==mod]['rppa'].values<2:
                class2others.append(p)

df['Panther_class']=np.where(df['Panther_class'].isin(class2others), 'Others', df['Panther_class'])
df=df.loc[df['protein'] !='Acetyl']


# Group by 'protein', 'class', and 'diff_word' and calculate the counts
df_plot=df[df.diff_word=='MethImproved']
grouped = df_plot.groupby(['protein', 'Panther_class']).size().unstack(fill_value=0).T
grouped['sum']=grouped.sum(axis=1)
grouped['percTotal']=(grouped['Total']/grouped['sum'])*100
grouped['percPhosph']=(grouped['Phosph']/grouped['sum'])*100
grouped=grouped.sort_values(by=['percTotal', 'percPhosph', 'sum'], ascending=[False, False, False])
grouped['all']=100
grouped['name']=grouped.index.astype(str)+' (' + grouped['sum'].astype(str) + ')'
grouped=grouped[grouped.index!='Others']

sns.barplot(data=grouped, x='name', y='all', color='deepskyblue')
sns.barplot(data=grouped, x='name', y='percTotal', color='orange')

plt.xticks(rotation=45, ha='right')
plt.ylabel('Proportion')
plt.tight_layout()
# plt.savefig(dir_output + "StackedBar_" + "percentImprovedRNA2RNAplusDNAm_proteinGroup_methyImproved" + '.png', format='png', dpi=600, transparent=True)
# plt.savefig(dir_output + "StackedBar_" + "percentImprovedRNA2RNAplusDNAm_proteinGroup_methyImproved" + '.pdf', format='pdf', dpi=600, transparent=True)
# plt.close()

##------Figure 2C------

PromoterCor=pd.read_csv('/home/CBBI/tsaih/Research_DNAm/Correlation_DNAmethRNA/'+
                        'Table_correlation_protein2RNA2DNAmeth_Promoters' + '.txt', sep='\t')
df=pd.read_csv(dir_output+'Table_proteingroups_forManualassignment.txt', sep='\t')
df=df[df.diff_word=='MethImproved']

df_plot=pd.merge(df, PromoterCor, left_on='rppa', right_on='proName')
df_plot=df_plot.reset_index()

o_list=[]
#counts follow central dogma?
for i in range(len(df_plot)):
    pval_g2p = df_plot['adjP_Value_g2p_bh'][i]
    pval_m2g = df_plot['adjP_Value_m2g_bh'][i]
    Corr_g2p = df_plot['Correlation_g2p'][i]
    Corr_m2g = df_plot['Correlation_m2g'][i]

    if pval_g2p < 0.05:
        g2p = '+' if Corr_g2p > 0 else '-'
    else:
        g2p = 'ns'

    if np.isnan(pval_m2g):
        m2g = 'np'
    elif pval_m2g < 0.05:
        m2g = '+' if Corr_m2g > 0 else '-'
    else:
        m2g = 'ns'

    o="g2p"+g2p+"m2g"+m2g
    o_list.append(o)

df_plot['CentralDgroup'] = o_list


grouped = pd.DataFrame(df_plot.groupby('CentralDgroup').count()['group_centraldogma'])
grouped['name']=grouped.index+' ('+grouped['group_centraldogma'].astype(str)+")"
order=['g2p+m2g-', 'g2p+m2g+', 'g2p+m2gns', 'g2p-m2g-', 'g2p-m2g+', 'g2pnsm2g-', 'g2pnsm2g+']

grouped=grouped.loc[order]
sns.barplot(data=grouped, x='name', y='group_centraldogma')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Counts')
plt.tight_layout()
# plt.savefig(dir_output + "Bar_" + "proteinobeyCentralDogma_methyImproved" + '.png', format='png', dpi=600, transparent=True)
# plt.savefig(dir_output + "Bar_" + "proteinobeyCentralDogma_methyImproved" + '.pdf', format='pdf', dpi=600, transparent=True)
# plt.close()

