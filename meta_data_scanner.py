# This is a sample Python script.


import os
import numpy as np
import pandas as pd


def composerStats(meta_data_file):
    df = pd.read_csv( meta_data_file )
    pd.options.display.width= None

    pd.set_option('display.max_rows', 3000)
    pd.set_option('display.max_columns', 3000)
    #print(df)
    df1 = df[ (df.use == 't') | (df.use == 'v')]
    print( f'labeled npy file count {df1.shape[0]} ')
    df2 = df.groupby(by=['composer'])['length_in_measures'].sum()
    df3 = df1.groupby(by=['composer','use'])['length_in_measures'].sum()
    print(df2)
    print(df3)


# scan through the folder of encoded .npy file to generate the meta data .csv file
def generateMetaData( encoded_data_folder ):
    df = pd.DataFrame(columns =['composer','title', 'npy_file_path', 'file_size', 'length_in_measures', 'sample_rate', 'use'])
    print( df )
    for root, dirs, files in os.walk( encoded_data_folder ):
        folder_elements =  root.replace("\\", "/").split('/')
        if len( folder_elements) > 2: composer = folder_elements[2]
        for file in files:
            if file.endswith('.npy'):
                np_file = os.path.join(root, file)
                print( f'scanning file: {np_file} size: {int(os.path.getsize( np_file )/1028)} kbyte')
                encoded_measures = np.load(np_file)
                base_name = os.path.basename(file)
                title = os.path.splitext(base_name)[0]
                print(f' composer: {composer}, title: {title}, path: {np_file}, length {encoded_measures.shape[0]} ')
                df.loc[len(df.index)] = [composer, title, np_file, int(os.path.getsize( np_file )/1024), encoded_measures.shape[0],None,None]

    df.to_csv( os.path.join( encoded_data_folder, 'meta_data.csv'), encoding='utf-8', index=False)


#generateMetaData("Dataset/classifier/")
LABLED_META_DATA_FILE = "Dataset/classifier/meta_data_labeled.csv"
composerStats( LABLED_META_DATA_FILE)