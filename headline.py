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
    print(Y)
    # Pad X and convert Y to categorical (Y consisted of integers)
    X = tf.keras.utils.pad_sequences(X, maxlen=maxlen)
    Y = to_categorical(Y, num_classes=max_features)

    return X, Y, tokenizer

max_features, max_len = 3500, 20
X, Y, tokenizer = format_data(df, max_features, max_len)

epochs = 100

model = Sequential()

# Embedding and GRU
model.add(Embedding(max_features, 300))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(30)))

# Output layer
model.add(Dense(max_features, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=epochs, batch_size=128, verbose=1)
'''
max_features = 3500

def get_sequence_of_tokens(corpus):
    # get tokens
    filters = "!\"#$%&()*+,-./;<=>?@[\\]^_`{|}~\t\n"
    tokenizer = Tokenizer(num_words=max_features, filters=filters)
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # convert to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
    input_sequences.append(n_gram_sequence)

    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(df['text'])
print("total words", total_words)
def generate_padded_sequences(input_sequences):
  max_sequence_len = max([len(x) for x in input_sequences])
  input_sequences = np.array(tf.keras.utils.pad_sequences(input_sequences,
                            maxlen=max_sequence_len, padding='pre'))
  predictors, label = input_sequences[:,:-1], input_sequences[:, -1]
  label = ku.to_categorical(label, num_classes = total_words)
  return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()

    # Add Input Embedding Layer
    model.add(Embedding(total_words,10, input_length=input_len))

    # Add Hidden Layer 1 — LSTM Layer
    model.add(Bidirectional(LSTM(30)))
    model.add(Dropout(0.1))

    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics=['accuracy'])

    return model

model = create_model(max_sequence_len, total_words)
model.fit(predictors, label, epochs=20, verbose=5)
'''