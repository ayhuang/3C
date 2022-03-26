
import numpy as np
import pandas as pd
import encoder

META_CVS_PATH = "Dataset/classifier/meta_data_labeled.csv"
MEASURES_PER_SAMPLE = 5
GAP_BETWEEN_SAMPLE = 2
#np.load('Dataset/classifier/Haydn/Miniature Concerto for Piano and Orchestra.npy' )

# find the last non zero col
def last_non_zero_col( arr ):
   # print(f' measure array shape {arr.shape}')
    max_col_idx = 0
    measure_idx = 0
    widest_measure = None
    for k in range( 0, arr.shape[0]):
        m = arr[k]  # a single measure, (100,2)
        non_zero = m.nonzero()
        largest_non_zero_idx = max(non_zero[0][len(non_zero[0])-1], non_zero[1][len(non_zero[1])-1])
        if max_col_idx < largest_non_zero_idx:
            max_col_idx = largest_non_zero_idx
            measure_idx = k
            widest_measure = m

    return measure_idx, max_col_idx, widest_measure

# loan data based on meta_data_file, return x_train, y_train, x_val, y_val
def find_padding( meta_data_file, measures_per_sample = MEASURES_PER_SAMPLE, max_skip_rate = 2 ):
    df = pd.read_csv( meta_data_file )
    pd.options.display.width= None

    pd.set_option('display.max_rows', 3000)
    pd.set_option('display.max_columns', 3000)
    df1 = df[ (df.use == 't') | (df.use == 'v')]
    print( f'total measure count {df1.shape[0]} ')

    non_zero_col = 0
    largest_measure = None
    for index, row in df1.iterrows():
        composer = row['composer']
        npy_file = row['npy_file_path']
        encoded_measure_array = np.load(npy_file)
        (m_idx, max_non_zero_col, widest_measure) = last_non_zero_col(encoded_measure_array)
        print( f' file {npy_file}, largest measure: {m_idx}, width: {max_non_zero_col}')
        if max_non_zero_col > non_zero_col:
            non_zero_col = max_non_zero_col
            largest_measure = widest_measure

    return non_zero_col, largest_measure


(non_zero_col, largest_measure ) = find_padding(META_CVS_PATH)

print(f' max non zero col: {non_zero_col}, largest measure {largest_measure}')

encoder.decodeFromArray( largest_measure ).show()

