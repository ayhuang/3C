# This is a sample Python script.


from encoder import *

import os
import numpy as np

midi_file = 'Dataset\classifier\Beethoven\Symphonies\\Symphony n7 3mov.mid'
#strm  = converter.parse(midi_file)
# strm.show()
# k = strm.parts[0].measure(1).keySignature
# measures = strm.getElementsByClass(stream.Measure)
# for m in measures:
#     print(f'found measure, {m.measureNumber}')
#
# newStream = transpose2C( strm, k)
# newStream.show()

def processSingleMidiFile( midi_file ):
    base_name = os.path.basename(midi_file)
    npfile_path = os.path.join( os.path.dirname( midi_file ), os.path.splitext(base_name)[0] + '.npy')

    encoded_array =  encodeFromMidi( midi_file, True)
    print(f' encoded {midi_file} array {encoded_array.shape}')
    np.save(  npfile_path, encoded_array)

processSingleMidiFile( midi_file  )
#MIDI_DATASET_FOLDER = "C:\\users\\ayh55\\Dataset"



