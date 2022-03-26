# This is a sample Python script.


import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

META_CVS_PATH = "Dataset/classifier/meta_data_labeled.csv"
MEASURES_PER_SAMPLE = 5
GAP_BETWEEN_SAMPLE = 2

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

#Two seperate embedding layers, one for tokens, one for token index (positions).

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
  #      self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.width = maxlen

    def call(self, x):
        #maxlen = tf.shape(x)[0].shape
        positions = tf.range(start=0, limit=self.width, delta=1)
        positions = self.pos_emb(positions)
#        print(f'position shape : {positions.shape}, value {positions[0:15]}')
  #      x = self.token_emb(x)
 #       print( f' token shape : {x.shape}')
        return x + positions


def encode_composer( name ):
    composers = ['Bach', 'Beethoven', 'Brahms', 'Chopin', 'Haendel', 'Haydn', 'Mendelssohn', 'Mozart', 'Schubert', 'Vivaldi']
    return composers.index( name )

#
#def normalize_pitch( x ):



# loan data based on meta_data_file, return x_train, y_train, x_val, y_val
def load_data( meta_data_file, measures_per_sample = MEASURES_PER_SAMPLE, max_skip_rate = 2 ):
    df = pd.read_csv( meta_data_file )
    pd.options.display.width= None

    #pd.set_option('display.max_rows', 3000)
    #pd.set_option('display.max_columns', 3000)
    df1 = df[ (df.use == 't') | (df.use == 'v')]
    print( f'npy file count {df1.shape[0]} ')
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
        #print( f' read {measures_in_file} measures from {npy_file}')
        k1 = 0
        k2 = k1 + measures_per_sample
        while k2 < measures_in_file:
            if row['use'] == 'v':
                y_val.append( encode_composer( composer ))
                x_val.append( encoded_measure_array[k1:k2])
            if row['use'] == 't':
                y_train.append( encode_composer( composer ))
                x_train.append( encoded_measure_array[k1:k2])
            #skip random number of measures
            k1 = k2 + random.randint(0, max_skip_rate)
            k2 = k1 + measures_per_sample

    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)


x_train, y_train, x_val, y_val = load_data(META_CVS_PATH)

print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")


"""## Create classifier model using transformer layer

Transformer layer outputs one vector for each time step of our input sequence.
Here, we take the mean across all time steps and
use a feed forward network on top of it to classify text.
"""

embed_dim = 200  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 5  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(MEASURES_PER_SAMPLE, embed_dim))
embedding_layer = TokenAndPositionEmbedding(MEASURES_PER_SAMPLE, 0, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

"""## Train and Evaluate"""

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.summary()

history = model.fit( x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))

