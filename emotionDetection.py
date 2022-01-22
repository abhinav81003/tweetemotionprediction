import re
import numpy as np
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib as plt
from sklearn.utils import shuffle
import itertools
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers

#importing the dataset 
dataset = pd.read_csv("https://d78e75dd-39c2-4f2d-a305-c95054bc5706.filesusr.com/ugd/77f91c_44a7732d95524cc6872176914dac3195.csv?dn=data.csv")
dataset = shuffle(dataset)
dataset = dataset.drop('author',1) 
dataset = dataset[1:900000]
#making all tweets lowercase
tweets = []
for tweet in dataset['tweet']:
  tweets.append(tweet.lower())
valence = np.array((dataset['valence']))
#pre processing and creating a word bag
class CreateWordBag:
  def __init__(self, text):
    self.text = text
    self.wordBag = set()
    self.wordToIndex = {}
    self.indexToWord = {}
    self.createIndices()

  def createIndices(self):
    for word in self.text:
      #creating the word bag by splitting the text into words
      self.wordBag.update(word.split(' '))

    #sorting the word bag
    self.wordBag = sorted(self.wordBag)

    #word to index map and index to word map
    for index, word in enumerate(self.wordBag):
      self.wordToIndex[word] = index
      self.indexToWord[index] = word

allText = CreateWordBag(tweets)

#converting the input tweets to input vectors corresponding to word's index in allText
inputs = []
for tweet in tweets:
  currentVector = []
  for word in tweet.split(' '):
    currentVector.append(allText.wordToIndex[word])
  inputs.append(np.array(currentVector))

# padding the input and output tensor to the maximum number of words possible in a tweet (140)
max_length_inp = 140
input_tensor = tf.keras.preprocessing.sequence.pad_sequences(inputs, 
                                                             maxlen=max_length_inp,
                                                             padding='post')

# one hot encoding the labels (Valence)
valence = list(set(dataset.valence.unique()))
num_valence = len(valence)
data_labels =  [set(vals) & set(valence) for vals in dataset[['valence']].values]

mlb = preprocessing.MultiLabelBinarizer()
bin_valence = mlb.fit_transform(data_labels)
target_tensor = np.array(bin_valence.tolist())

# creating a method to get the actual valence 
valence_dict = {0: 'negative', 1: 'positive'} #change this if 3 valences work

def getValence(vector):
  index = np.argmax(vector)
  return valence_dict[index]

#splitting data
inputs = np.array(inputs,dtype=object)
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.4)
input_tensor_val, input_tensor_test, target_tensor_val, target_tensor_test = train_test_split(input_tensor_val, target_tensor_val, test_size=0.5)

#using data loaders to easily manipulate data
# TRAIN_BUFFER_SIZE = len(input_tensor_train)
# VAL_BUFFER_SIZE = len(input_tensor_val)
# TEST_BUFFER_SIZE = len(input_tensor_test)
# BATCH_SIZE = 64
# TRAIN_N_BATCH = TRAIN_BUFFER_SIZE // BATCH_SIZE
# VAL_N_BATCH = VAL_BUFFER_SIZE // BATCH_SIZE
# TEST_N_BATCH = TEST_BUFFER_SIZE // BATCH_SIZE

embedding_dim = 256
units = 1024
vocab_inp_size = len(allText.wordToIndex)
target_size = num_valence

from keras.models import Sequential
from keras.layers import Embedding, Input
from keras.layers.merge import Concatenate
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Dropout, concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import metrics
from keras.models import Model
import pickle

model = Sequential()

# First LSTM layer (return sequence so that we can feed the output into the 2nd LSTM layer)
model.add(LSTM(64,input_shape=input_tensor.shape, return_sequences = True, activation = 'relu'))
model.add(Dropout(.2))

# Second LSTM layer 
# Don't return sequence this time, because we're feeding into a fully-connected layer
model.add(LSTM(64, activation='relu'))
model.add(Dropout(.2))

# Dense 1
model.add(Dense(32, activation='relu'))
model.add(Dropout(.2))

# Dense 2 (final vote)
model.add(Dense(1, activation = 'sigmoid'))
model.add(Flatten())

######################################

LOSS = 'binary_crossentropy' # because we're classifying between 0 and 1
OPTIMIZER = 'RMSprop' # RMSprop tends to work well for recurrent models

model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = [metrics.binary_accuracy])

TEST_SIZE = 0.4

#####################################

EPOCHS = 5
BATCH_SIZE = 2058

#####################################

model.fit(input_tensor, target_tensor, 
          epochs = EPOCHS,
          batch_size = BATCH_SIZE)
