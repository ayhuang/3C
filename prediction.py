# inference from saved model


import numpy as np
import preprocessor
from tensorflow import keras
from sklearn import preprocessing

MODEL_FOLDER = "Model"
#INPUT_SAMPLE_NPY = "Dataset/classifier/Bach/AveMaria.npy"
#INPUT_SAMPLE_NPY = "Dataset/classifier/Bach/Prelude and Fugue in A, BWV 888.npy"
#INPUT_SAMPLE_NPY = "Dataset/classifier/Beethoven/Symphonies/Symphony op93 n8 3mov .npy"
#INPUT_SAMPLE_NPY = "Dataset/classifier/Mozart/K314 Flute Concerto n2 2mov.npy"
#INPUT_SAMPLE_NPY = "Dataset/classifier/Mozart/K617 Adagio.npy"
#INPUT_SAMPLE_NPY = "Dataset/classifier/Mozart/Symphonies/Symphony n33 K319 2mov.npy"
INPUT_SAMPLE_NPY = "Dataset/classifier/Rachmaninov/rach3.npy"

model = keras.models.load_model( MODEL_FOLDER )

model.summary()

samples = preprocessor.load_single_piece(INPUT_SAMPLE_NPY, preprocessor.MEASURES_PER_SAMPLE, preprocessor.GAP_BETWEEN_SAMPLE)
x = np.array( samples )[:,:,:-1]
pred = model.predict( x ).sum( axis=0 )/x.shape[0]
le = preprocessing.LabelEncoder()
le.fit( preprocessor.COMPOSERS_LIST)
composrs = le.inverse_transform([0,1,2,3,4,5])
sorted_result = sorted( dict( zip(composrs, pred)).items(), key=lambda x:x[1], reverse=True)
print( sorted_result )