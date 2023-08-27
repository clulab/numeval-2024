import pandas as pd
import string
import numpy as np
import json
import re
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import tensorflow as tf
tf.random.set_seed(2)
from keras.utils import to_categorical
from numpy.random import seed
seed(1)
from keras.utils import pad_sequences

f = open('DryRun_Headline_Generation.json')
df = pd.read_json(f)

df['news'] = df['news'].apply(lambda x: re.sub(r'\([^)]*\)', '', x))
df['text'] = df[['news', 'headline']].apply(" ".join, axis=1)
print(df['text'].head())

START = '÷'
END = '■'


def format_data(data, max_features, maxlen, shuffle=False):

    # Add start and end tokens
    data['headline'] = START + ' ' + data['text'].str.lower() + ' ' + END

    text = data['headline']

    # Tokenize text
    filters = "!\"#$%&()*+,-./;<=>?@[\\]^_`{|}~\t\n"
    tokenizer = Tokenizer(num_words=max_features, filters=filters)
    tokenizer.fit_on_texts(list(text))
    corpus = tokenizer.texts_to_sequences(text)

    # Build training sequences of (context, next_word) pairs.
    # Note that context sequences have variable length. An alternative
    # to this approach is to parse the training data into n-grams.
    X, Y = [], []
    for line in corpus:
        for i in range(1, len(line) - 1):
            X.append(line[:i + 1])
            Y.append(line[i + 1])
    # Pad X and convert Y to categorical (Y consisted of integers)
    X = tf.keras.utils.pad_sequences(X, maxlen=maxlen)
    Y = to_categorical(Y, num_classes=max_features)

    return X, Y, tokenizer

max_features, max_len = 3500, 20
X, Y, tokenizer = format_data(df, max_features, max_len)

epochs = 50

model = Sequential()

# Embedding and GRU
model.add(Embedding(max_features, 300))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128)))

# Output layer
model.add(Dense(max_features, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=epochs, batch_size=128, verbose=1)
