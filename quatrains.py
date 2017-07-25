#!/usr/bin/env python3

# See also Chapters 6, 8 of "Deep Learning with Python" by Francois Chollet (Manning)
# https://www.manning.com/books/deep-learning-with-python

# Cloudera Data Science Workbench: install tensorflow and keras if not already.
# Install tensorflow-gpu instead if using GPUs.
'''
!pip3 install -U tensorflow tensorboard keras
'''

import math
import numpy as np
import random
import tensorflow as tf

from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Dense, Embedding, GRU
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2

# Configuration parameters.
# Whether or not to configure for a GPU
use_gpu = False
# Dimension of existing GLoVe embedding to use; must be 50, 100, 200, 300
embedding_dim = 300
# Dimension of recurrent GRU below
gru_dim = 128
# Dropout applied to GRU
dropout = 0.5
# Regularization factor
regularization = 0.02
# Training batch size
batch_size = 512
# RMSProp learning rate
learning_rate = 0.003
# Training will take sequences of this many words and learn to predict the following word
phrase_len = 4

# Load the text of Nostradamus's 942 quatrains, 1 per line
quatrains = []
quatrains_file = open('quatrains.txt')
for line in quatrains_file:
    quatrains.append(line)
quatrains_file.close()
# Shuffle them for good measure
random.shuffle(quatrains)

# Use a Keras Tokenizer to turn text into tokens, and assign tokens to unique indices
tokenizer = Tokenizer(filters='"#$%&()*+/:<=>@[\\]^_`{|}~\t\r\n', lower=True)
tokenizer.fit_on_texts(quatrains)
# Maps words to index, as you'd expect
word_index = tokenizer.word_index
# Inverse mapping:
index_word = {index: word for word, index in word_index.items()}
num_distinct_words = len(word_index)

# Weight words in the loss function such that more common words are given more importance.
# The weight here is, somewhat arbitrarily, the square root of the count.
# This _tends_ to make the output a little more normal-sounding.
word_index_weights = {word_index[word]: math.sqrt(count) for word, count in tokenizer.word_counts.items()}
word_index_weights[0] = 1.0

# + 1 embeddings because 0 is unused by Tokenizer, and we need an extra embedding to represent
# the end of a quatrain. Random values are generated to start, which will mostly be overwritten by the
# embedding. The ones that remain will then start with small random embeddings.
embeddings = np.random.rand(num_distinct_words + 1, embedding_dim)

# Read through embedding file and copy in any embedding for words that appear in the corpus
embeddings_file = open('glove.6B.{}d.txt'.format(embedding_dim))
for line in embeddings_file:
    tokens = line.split()
    word = tokens[0]
    if word in word_index:
        embeddings[word_index[word]] = np.asarray(tokens[1:], dtype='float32')
embeddings_file.close()

phrases = []
next_word_encodings = []
for sequence in tokenizer.texts_to_sequences(quatrains):
    # add an 'end' marker to the end, and start, of all sequences
    sequence = [0] + sequence + [0]
    # Pick any location that can start a phrase of phrase_len words and still have an element after
    for start_index in range(0, len(sequence) - phrase_len):
        next_index = start_index + phrase_len
        # Append the subsequence as inputs
        phrases.append(sequence[start_index : next_index])
        # Append the next word, one-hot-encoded. Only one of these zeroes will be set to 1
        next_word_encoding = np.zeros(num_distinct_words + 1, dtype=np.bool)
        next_word_encoding[sequence[next_index]] = 1
        next_word_encodings.append(next_word_encoding)

if use_gpu:
    # Configure TensorFlow to not grab all available GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    K.set_session(session)
    device = '/gpu:0'
else:
    device = '/cpu:0'


# Configure a TensorFlow model via Keras
model = Sequential()
# Use potentially different devices to compute one part of the model vs another
with tf.device(device):
    # Embedding layer converts words (indices) into embeddings
    # The actual embedding matrix is set below
    model.add(Embedding(num_distinct_words + 1, embedding_dim, input_length=phrase_len, name="embedding"))
    # Learn a recurrent model of sequences of words with a Gated Recurrent Unit (GRU)
    model.add(GRU(gru_dim,
                  name="GRU",
                  dropout=dropout,
                  recurrent_dropout=dropout,
                  kernel_regularizer=l2(regularization),
                  bias_regularizer=l2(regularization),
                  recurrent_regularizer=l2(regularization)))
    # Predict one of the possible words (or end) with standard dense layer plus softmax activation
    model.add(Dense(num_distinct_words + 1, name="dense", activation='softmax'))

# Actually set the weights for the embedding
# Note that this is not 'frozen' and is left trainable
model.layers[0].set_weights([embeddings])
# Compile the model with appropriate loss for multiclass classification
model.compile(optimizer=RMSprop(lr=learning_rate), loss='categorical_crossentropy', metrics=['acc'])
# Print a summary
model.summary()

# Need to work with these values as Numpy arrays
phrases = np.array(phrases)
# Shuffle the data, by computing a permutation of its indices
shuffled_indices = np.random.permutation(len(phrases))
# Most of the data will be used for training; take most of the shuffled indices as random training set,
# and remainder of indices indicate the validation set
training_size = int(0.95 * len(phrases))
train_indices = shuffled_indices[:training_size]
val_indices = shuffled_indices[training_size:]

# Split input into train/validation
phrases_train = phrases[train_indices]
phrases_val = phrases[val_indices]

# Split outputs in exactly the same way
next_word_embeddings = np.array(next_word_encodings)
next_word_embeddings_train = next_word_embeddings[train_indices]
next_word_embeddings_val = next_word_embeddings[val_indices]

for run in range(0, 1000):
    print('Run {}'.format(run))

    tensorboard = TensorBoard(log_dir="logs/{}".format(run), write_graph=True, histogram_freq=5)

    # Train for just a few epochs
    model.fit(phrases_train,
              next_word_embeddings_train,
              class_weight=word_index_weights,
              epochs=5,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(phrases_val, next_word_embeddings_val),
              verbose=2,
              callbacks=[tensorboard])

    # Generate a few random outputs from the model so far
    print()
    for i in range(0, 5):
        # Begin with a dummy 'phrase' of preceding text (all 0 / end markers)
        random_phrase = np.array([0] * phrase_len)
        # Build up a quatrain word by word by predicting the next word
        emitted_quatrain = []
        # Cap the size of the emitted quatrain in case it runs on a while
        while len(emitted_quatrain) < 32:
            # Need to reshape the input to use with model.predict
            random_phrase_t = np.copy(random_phrase)
            random_phrase_t.shape = (1, phrase_len)
            # Predict the next word, to get a distribution over all words
            pred_next_word = model.predict(random_phrase_t)[0]
            # Choose one word from this distribution; not always the most likely word!
            draw = np.random.uniform()
            for pred_next_word_index in range(0, len(pred_next_word)):
                draw -= pred_next_word[pred_next_word_index]
                if draw < 0.0:
                    break
            if pred_next_word_index == 0:
                # End of quatrain
                break
            else:
                # Append the predicted next word
                emitted_quatrain.append(index_word[pred_next_word_index])
                # Update the phrase to drop first word, add next new word at the end
                random_phrase = np.append(random_phrase[1:], pred_next_word_index)

        # Print a generated snippet for this run
        print(" ".join(emitted_quatrain))

    print()
