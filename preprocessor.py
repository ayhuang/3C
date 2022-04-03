#preprocessing data for training

import os
import random
import numpy as np
import pandas as pd

MEASURES_PER_SAMPLE = 5
GAP_BETWEEN_SAMPLE = 1

COMPOSERS_LIST = ['Bach', 'Beethoven', 'Haydn', 'Mozart', 'Schubert', 'Other']
def encode_composer( name ):

    if name in COMPOSERS_LIST:
        return COMPOSERS_LIST.index(name)
    else:
        return 5  # other


# the measure array has the shape (none, 100, 2)
def normalize_measure_array( x ):
    a = np.empty( x.shape, dtype='float32')
    # the pitch ranges from 21 to 108, middle C is 64, note rest is marked as -1
    a[:,:,0] = x[:,:,0]/88
    # assuming time duration ranges from 1 to 16
    a[:,:,1] = x[:,:,1]/16
    return a


# load a single npy file, return an array of samples
def load_single_piece( npy_file, measures_per_sample, max_skip_rate = 2, composer_label=-1):
    encoded_measure_array = normalize_measure_array(np.load(npy_file))
    measures_in_file = encoded_measure_array.shape[0]
    print( f' read {measures_in_file} measures from {npy_file}')

    samples = []
    k1 = 0
    k2 = k1 + measures_per_sample
    while k2 < measures_in_file:
        # make one sample
        s = encoded_measure_array[k1:k2]  # shape = (measures_per_sample, max_depth_per_measure, 2)
        # reshape each single measure from (max_depth_per_measure,2) to a vector of ( 2*max_depth_per_measure)
        s = s.reshape(measures_per_sample, s.shape[1] * s.shape[2])
        # for each measure add composer label as the last column
        x = np.full((measures_per_sample, s.shape[1] + 1), composer_label, dtype='float32')
        x[:, :-1] = s
        samples.append(x)
        # skip random number of measures
        k1 = k2 + random.randint(0, max_skip_rate)
        k2 = k1 + measures_per_sample
    return samples


# loan data based on meta_data_file, return x_train, y_train, x_val, y_val
def load_data( meta_data_file,  measures_per_sample, shuffle=True,max_skip_rate = 2 ):
    df = pd.read_csv( meta_data_file )
    pd.options.display.width= None

    #pd.set_option('display.max_rows', 3000)
    #pd.set_option('display.max_columns', 3000)
    df1 = df[ (df.use == 't') | (df.use == 'v')]
    print( f'npy file count {df1.shape[0]} ')
    #df2 = df1.groupby(by=['composer','use'])['length_in_measures'].sum()

    s_val = [];  s_train = []
    # a sample is composed of "measures_per_sample" of measures,
    # between two samples, skip a random number of measures
    for index, row in df1.iterrows():
        composer = row['composer']
        npy_file =  row['npy_file_path']
        s_train.append( load_single_piece( npy_file, measures_per_sample, max_skip_rate, encode_composer(composer)))

    t_arr = np.array(s_train)
    #val_arr = np.array(s_val)
    if shuffle:
         np.random.shuffle( t_arr )
         index = int(t_arr.shape[0]*0.90)
         #np.random.shuffle( val_arr )

    # split
    train_arr = t_arr[:index]
    val_arr = t_arr[index:]
    #separate label, input
    x_train = train_arr[:,:,:-1]
    x_val = val_arr[:,:,:-1]
    y_train = train_arr[:,1,-1].reshape(-1)
    y_val = val_arr[:,1,-1].reshape(-1)
    return (x_train, y_train),(x_val, y_val)

