# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:01:08 2021

@author: putta
"""

import numpy as np
import os
import csv
from random import random, sample, seed

data_path = 'dataDL.csv'
embeddings_path = 'glove.6B.50d.txt'

titles = []
hours = []
minutes = []
dayofweeks = []
dayofyears = []
is_top_submission = []

with open(data_path, 'r', encoding="latin-1") as f:
    reader = csv.DictReader(f)
    for submission in reader:
        titles.append(submission['title'])
        hours.append(submission['hour'])
        minutes.append(submission['minute'])
        dayofweeks.append(submission['dayofweek'])
        dayofyears.append(submission['dayofyear'])
        is_top_submission.append(submission['is_top_submission'])
            
titles = np.array(titles)
hours = np.array(hours, dtype=int)
minutes = np.array(minutes, dtype=int)
dayofweeks = np.array(dayofweeks, dtype=int)
dayofyears = np.array(dayofyears, dtype=int)
is_top_submission = np.array(is_top_submission, dtype=int)

print(titles[0:2])
print(titles.shape)
print(hours[0:2])
print(minutes[0:2])
print(dayofweeks[0:2])
print(dayofyears[0:2])
print(is_top_submission[0:2])


1 - np.mean(is_top_submission)

from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence, Tokenizer

max_features = 40000

word_tokenizer = Tokenizer(max_features)
word_tokenizer.fit_on_texts(titles)

print(str(word_tokenizer.word_counts)[0:100])
print(str(word_tokenizer.word_index)[0:100])
print(len(word_tokenizer.word_counts))   # true word count


titles_tf = word_tokenizer.texts_to_sequences(titles)

print(titles_tf[0])

titles_tf = word_tokenizer.texts_to_sequences(titles)

print(titles_tf[0])
maxlen = 20
titles_tf = sequence.pad_sequences(titles_tf, maxlen=maxlen)

print(titles_tf[0])

embedding_vectors = {}

with open(embeddings_path, 'r', encoding="latin-1") as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        word = line_split[0]
        embedding_vectors[word] = vec
        
print(embedding_vectors['you'])

weights_matrix = np.zeros((max_features + 1, 50))

for word, i in word_tokenizer.word_index.items():

    embedding_vector = embedding_vectors.get(word)
    if embedding_vector is not None and i <= max_features:
        weights_matrix[i] = embedding_vector

# index 0 vector should be all zeroes, index 1 vector should be the same one as above
print(weights_matrix[0:2,:])

dayofyears_tf = dayofyears - 1

print(dayofyears_tf[0:10])
      

# MODEL BUILDING

from keras.models import Input, Model
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, concatenate, Activation
from keras.layers.core import Masking, Dropout, Reshape
from keras.layers.normalization import BatchNormalization

batch_size = 32
embedding_dims = 50
epochs = 20

titles_input = Input(shape=(maxlen,), name='titles_input')
titles_embedding = Embedding(max_features + 1, embedding_dims, weights=[weights_matrix])(titles_input)
titles_pooling = GlobalAveragePooling1D()(titles_embedding)

aux_output = Dense(1, activation='sigmoid', name='aux_out')(titles_pooling)

meta_embedding_dims = 64

hours_input = Input(shape=(1,), name='hours_input')
hours_embedding = Embedding(24, meta_embedding_dims)(hours_input)
hours_reshape = Reshape((meta_embedding_dims,))(hours_embedding)

dayofweeks_input = Input(shape=(1,), name='dayofweeks_input')
dayofweeks_embedding = Embedding(7, meta_embedding_dims)(dayofweeks_input)
dayofweeks_reshape = Reshape((meta_embedding_dims,))(dayofweeks_embedding)

minutes_input = Input(shape=(1,), name='minutes_input')
minutes_embedding = Embedding(60, meta_embedding_dims)(minutes_input)
minutes_reshape = Reshape((meta_embedding_dims,))(minutes_embedding)

dayofyears_input = Input(shape=(1,), name='dayofyears_input')
dayofyears_embedding = Embedding(366, meta_embedding_dims)(dayofyears_input)
dayofyears_reshape = Reshape((meta_embedding_dims,))(dayofyears_embedding)

merged = concatenate([titles_pooling, hours_reshape, dayofweeks_reshape, minutes_reshape, dayofyears_reshape])

hidden_1 = Dense(256, activation='relu')(merged)
hidden_1 = BatchNormalization()(hidden_1)

main_output = Dense(1, activation='sigmoid', name='main_out')(hidden_1)

# COMPILE THE MODEL

model = Model(inputs=[titles_input,
                      hours_input,
                      dayofweeks_input,
                      minutes_input,
                      dayofyears_input], outputs=[main_output, aux_output])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              loss_weights=[1, 0.2])

model.summary()

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.callbacks import CSVLogger

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

plot_model(model, to_file='model.png')
plot_model(model, to_file='model_shapes.png', show_shapes=True)

# TRAIN THE MODEL

seed(123)
split = 0.2

# returns randomized indices with no repeats
idx = sample(range(titles_tf.shape[0]), titles_tf.shape[0])

titles_tf = titles_tf[idx, :]
hours = hours[idx]
dayofweeks = dayofweeks[idx]
minutes = minutes[idx]
dayofyears_tf = dayofyears_tf[idx]
is_top_submission = is_top_submission[idx]

print(1 - np.mean(is_top_submission[:(int(titles_tf.shape[0] * split))]))

csv_logger = CSVLogger('training.csv')

model.fit([titles_tf, hours, dayofweeks, minutes, dayofyears_tf], [is_top_submission, is_top_submission],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=split, callbacks=[csv_logger])

# Make Prediction

def encode_text(text, maxlen):
    encoded = word_tokenizer.texts_to_sequences([text])
    return sequence.pad_sequences(encoded, maxlen=maxlen)


input_text = "Russia will soon have over 120,000 troops on Ukraine's borders, Kyiv says"
encoded_text = encode_text(input_text, maxlen)
print(encoded_text)

input_hour = np.array([02])
input_minute = np.array([15])
input_dayofweek = np.array([3])
input_dayofyear = np.array([217])

model.predict([encoded_text, input_hour, input_dayofweek, input_minute, input_dayofyear])

input_text = "Russia will have over 120,000 troops on Ukraine's borders, Kyiv says"
encoded_text = encode_text(input_text, maxlen)
model.predict([encoded_text, input_hour, input_dayofweek, input_minute, input_dayofyear])

input_text = "Russia will soon have over 120,000 troops on Ukraine's borders"
encoded_text = encode_text(input_text, maxlen)
model.predict([encoded_text, input_hour, input_dayofweek, input_minute, input_dayofyear])

input_text = "Russia will have troops on Ukraine's borders, Kyiv says”"
encoded_text = encode_text(input_text, maxlen)
model.predict([encoded_text, input_hour, input_dayofweek, input_minute, input_dayofyear])

import pandas as pd
import pylab as plt

# Create dataframe
pd.read_csv('training.csv', index_col='epoch')
