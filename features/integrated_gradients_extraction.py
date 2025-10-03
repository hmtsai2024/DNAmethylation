#!/usr/bin/env python
# coding: utf-8

import os
import re
import time
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from pathlib import Path
from tensorflow.keras.models import load_model
tf.compat.v1.disable_eager_execution()
import innvestigate
from import_data import load_data

datatype='DNAm'
dir_data='/home/CBBI/tsaih/data/DNAm/'

#-----Load data-----
print('Reading input data...')
rna_train, rna_labels_train, _, _ = load_data(dir_data + "RNA_train.txt")
rna_val, rna_labels_val, _, _ = load_data(dir_data + "RNA_val.txt")
rna_test, rna_labels_test, _, _ = load_data(dir_data + "RNA_test.txt")

dnam_train, dnam_labels_train, _, _ = load_data(dir_data + "DNAm_train.txt")
dnam_val, dnam_labels_val, _, _ = load_data(dir_data + "DNAm_val.txt")
dnam_test, dnam_labels_test, _, _ = load_data(dir_data + "DNAm_test.txt")

print('Reading output (protein) data...')
pro_train, _, _, gene_names_pro = load_data(dir_data + "Protein_train.txt")
pro_val, _, _, _ = load_data(dir_data + "Protein_val.txt")
pro_test, _, _, _ = load_data(dir_data + "Protein_test.txt")

#-----Setup input features-----
if datatype == 'merge':
    nfeatures = rna_train.shape[1] + dnam_train.shape[1]
    x_train = np.concatenate((rna_train, dnam_train), axis=1)
    x_val   = np.concatenate((rna_val, dnam_val), axis=1)
    x_test  = np.concatenate((rna_test, dnam_test), axis=1)
elif datatype == 'DNAm':
    nfeatures = dnam_train.shape[1]
    x_train, x_val, x_test = dnam_train, dnam_val, dnam_test
elif datatype == 'RNA':
    nfeatures = rna_train.shape[1]
    x_train, x_val, x_test = rna_train, rna_val, rna_test
    
#-----Reshape for model input-----
reshape_data = lambda x: x.reshape(x.shape[0], nfeatures, 1)
x_train, x_val, x_test = map(reshape_data, [x_train, x_val, x_test])
data = np.vstack((x_train, x_val, x_test))
del x_train, x_val, x_test

#-----Model path-----
if datatype == 'merge':
    dir_input = '/home/CBBI/tsaih/Research_DNAm/EarlymergeX_manual/'
    tmpName = 'DeepG2Pstructure_EarlymergeManual'
    model_file = dir_input + tmpName + '_model_8.h5'
elif datatype == 'RNA':
    dir_input = '/home/CBBI/tsaih/Research_DNAm/diffX/'
    tmpName = 'DeepG2Pstructure_RNA'
    model_file = dir_input + tmpName + '_model_2.h5'
elif datatype == 'DNAm':
    dir_input = '/home/CBBI/tsaih/Research_DNAm/diffX/'
    tmpName = 'DeepG2Pstructure_DNAm'
    model_file = dir_input + tmpName + '_model_6.h5'

dir_output = os.path.join(dir_input, f'IG_{datatype}/')
Path(dir_output).mkdir(parents=True, exist_ok=True)

#-----Load model-----
model_saved = load_model(model_file)
model_saved.summary()
for layer in model_saved.layers:
    print(layer)

#-----Check completed proteins-----
done_files= glob(dir_output + 'gradient_' + '*.txt')
done_pro=[]
for d in done_files:
    done_pro.append(re.search('gradient_' + '(.*).txt', d).group(1))
print(len(done_pro))

#-----Set up analyzer-----
analyzer_index= innvestigate.create_analyzer("integrated_gradients",
                                          model_saved,
                                          neuron_selection_mode="index")
print(len(gene_names_pro))

project = 'single'  # 'full' or 'single'
start_time = time.time()

def analyze_protein(protein, data, nfeatures, dir_output):
    idx = gene_names_pro.index(protein)
    print(f"{idx+1} : {protein}")
    if protein in done_pro:
        print("    done")
        return

    analysis_value_index = np.empty((data.shape[0], nfeatures))
    for j in tqdm(range(data.shape[0])):
        analysis_index = analyzer_index.analyze(data[j:j+1], neuron_selection=idx)
        analysis_value_index[j] = analysis_index.reshape(1, nfeatures)

    np.savetxt(
        os.path.join(dir_output, f'gradient_{protein}_test.txt'),
        analysis_value_index.T,
        delimiter='\t', fmt='%.10f'
    )

if project == 'full':
    for protein in gene_names_pro:
        analyze_protein(protein, data, nfeatures, dir_output)
else:
    analyze_protein('ERALPHA', data, nfeatures, dir_output) #CYCLINE2

print('END')

#-----Time-----
elapsed_time = time.time() - start_time
if elapsed_time >= 3600:
    print(f"Time taken: {elapsed_time/3600:.2f} hours")
elif elapsed_time >= 60:
    print(f"Time taken: {elapsed_time/60:.2f} minutes")
else:
    print(f"Time taken: {elapsed_time:.2f} seconds")

