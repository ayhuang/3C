# This is a sample Python script.


import os
import numpy as np
import pandas as pd

MIDI_DATASET_FOLDER = "..\\Dataset\\classifier"
df = pd.DataFrame(columns =['composer','title', 'npy_file_path', 'file_size', 'length_in_measures', 'sample_rate', 'use'])
print( df )
for root, dirs, files in os.walk( MIDI_DATASET_FOLDER ):
    folder_elements =  root.split('\\')
    if len( folder_elements) > 3: composer = folder_elements[3]
    for file in files:
        if file.endswith('.npy'):
            np_file = os.path.join(root, file)
      #      print( f'scanning file: {np_file} size: {int(os.path.getsize( np_file )/1028)} kbyte')
            encoded_measures = np.load(np_file)
            base_name = os.path.basename(file)
            title = os.path.splitext(base_name)[0]
  #          print(f' composer: {composer}, title: {title}, path: {np_file}, length {encoded_measures.shape[0]} ')
            df.loc[len(df.index)] = [composer, title, np_file, int(os.path.getsize( np_file )/1024), encoded_measures.shape[0],None,None]

df.to_csv( os.path.join( MIDI_DATASET_FOLDER, 'meta_data.cvs'), encoding='utf-8', index=False)