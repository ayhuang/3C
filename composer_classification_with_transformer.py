# This is a sample Python script.


import os
import random

import numpy as np
import pandas as pd

META_CVS_PATH = "../Dataset/classifier/meta_data_labeled.csv"
MEASURES_PER_SAMPLE = 5
GAP_BETWEEN_SAMPLE = 2
np.load('Dataset/classifier/Haydn/Miniature Concerto for Piano and Orchestra.npy' )

def encode_composer( name ):
    composers = ['Bach', 'Beethoven', 'Brahms', 'Chopin', 'Haendel', 'Haydn', 'Mendelssohn', 'Mozart', 'Schubert', 'Vivaldi']
    return composers.index( name )

# loan data based on meta_data_file, return x_train, y_train, x_val, y_val
def load_data( meta_data_file, measures_per_sample = MEASURES_PER_SAMPLE, max_skip_rate = 2 ):
    df = pd.read_csv( meta_data_file )
    pd.options.display.width= None

    pd.set_option('display.max_rows', 3000)
    pd.set_option('display.max_columns', 3000)
    df1 = df[ (df.use == 't') | (df.use == 'v')]
    print( f'total measure count {df1.shape[0]} ')
    #df2 = df1.groupby(by=['composer','use'])['length_in_measures'].sum()
    y_train = []
    x_train = []
    y_val = []
    x_val = []
    for index, row in df1.iterrows():
        composer = row['composer']
        npy_file =  row['npy_file_path']
        encoded_measure_array = np.load(npy_file)
        measures_in_file = encoded_measure_array.shape[0]
        print( f' read {measures_in_file} measures from {npy_file}')
        k1 = 0
        k2 = k1 + measures_per_sample
        while k2 < measures_in_file:
            if row['use'] == 'v':
                y_val.append( encode_composer( composer ))
                x_val.append( np.transpose(encoded_measure_array[k1:k2]))
            if row['use'] == 't':
                y_train.append( encode_composer( composer ))
                x_train.append( np.transpose(encoded_measure_array[k1:k2]))
            #skip random number of measures
            k1 = k2 + random.randint(0, max_skip_rate)
            k2 = k1 + measures_per_sample

    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)


x_train, y_train, x_val, y_val = load_data(META_CVS_PATH)

print(f' train array {x_train.shape}, {y_train.shape}, validation array {x_val.shape}, {y_val.shape}')

