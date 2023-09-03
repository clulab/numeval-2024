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
print(df.head())

f.close()

''' SHOULD I COMBINED NEWS AND HEALDLINE TO TRAIN A MODEL" '''
df['news'] = df['news'].apply(lambda x: re.sub(r'\([^)]*\)', '', x))
df['text'] = df[['news', 'headline']].apply(" ".join, axis=1)
print(df['text'].head())

START = '÷'
END = '■'

print("DF LEN", len(df['text']))

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

epochs = 30

model = Sequential()

# Embedding and GRU
model.add(Embedding(max_features, 300))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128)))

# Output layer
model.add(Dense(max_features, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=epochs, batch_size=128, verbose=1)

idx_to_words = {value: key for key, value in tokenizer.word_index.items()}


def sample(preds, temp=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    preds = np.exp(preds) / np.sum(np.exp(preds))
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

def process_input(text):
    tokenized_input = tokenizer.texts_to_sequences([text])[0]
    return pad_sequences([tokenized_input], maxlen=max_len - 1)


def generate_text(input_text, model, n=7, temp=1.0):
    if type(input_text) is str:
        sent = input_text
    else:
        sent = ' '.join(input_text)

    tokenized_input = process_input(input_text)

    while True:
        preds = model.predict(tokenized_input, verbose=0)[0]
        pred_idx = sample(preds, temp=temp)
        pred_word = idx_to_words[pred_idx]

        if pred_word == END:
            return sent

        sent += ' ' + pred_word
        #         print(sent)
        #         tokenized_input = process_input(sent[-n:])
        tokenized_input = process_input(sent)


#text = generate_text(START, model, temp=0.01)
#print(text[2:])

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1,  padding='pre')
        predicted = model.predict(token_list, verbose=0)
        classes = np.argmax(predicted, axis=1)
        output_word = ''
        for word,index in tokenizer.word_index.items():
            if index == classes:
                output_word = word
                seed_text += " "+output_word
    return seed_text.title()

print("Generated Headline")
print(generate_text("Walmart", 7, model, max_len))
print("Actual Headline")
print(df['headline'][0])