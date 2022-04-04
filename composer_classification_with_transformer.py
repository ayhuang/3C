# This is a sample Python script.


import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import preprocessor

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
        return x + positions


###############################################################################
META_CVS_PATH = "Dataset/classifier/meta_data_labeled.csv"
MEASURES_PER_SAMPLE = 10
GAP_BETWEEN_SAMPLE = 1
MODEL_FOLDER = "Model"

(x_train, y_train), (x_val, y_val) = preprocessor.load_data(META_CVS_PATH,measures_per_sample=MEASURES_PER_SAMPLE,max_skip_rate=GAP_BETWEEN_SAMPLE, shuffle=True)

print(f' {x_train.shape} Training sequences, with label {y_train.shape}')
print(f' {x_val.shape} Training sequences, with label {y_val.shape}')

embed_dim = 200  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 5  # Hidden layer size in feed forward network inside transformer
num_categories = len( preprocessor.COMPOSERS_LIST )

## Create classifier model using transformer layer

inputs = layers.Input(shape=(MEASURES_PER_SAMPLE, embed_dim))
embedding_layer = TokenAndPositionEmbedding(MEASURES_PER_SAMPLE, 0, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.3)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(200, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_categories, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

## Train and Evaluate

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.summary()

history = model.fit( x_train, y_train, batch_size=32, epochs=200, validation_data=(x_val, y_val), shuffle=True)

model.save(MODEL_FOLDER )
