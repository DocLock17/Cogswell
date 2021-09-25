#!/bin/bash/python3
# This is some clean up code that should ensure the script runs virtually anywhere


try:
  import simpleaudio as sa
except:
#   !pip install simpleaudio
    pass

try:
  import speech_recognition as sr 
except:
#   !pip install speechrecognition==3.8.1
  pass
try:
  import pyttsx3 
except:
#   !pip install pyttsx3
  pass

try:
  import nltk
except:
#   !pip install nltk
  pass

try:
  punkt_test = nltk.word_tokenize("test")
except:
  nltk.download('punkt')

try:
  test_lemmatizer = WordNetLemmatizer()
  test_lemmatizer.lemmatize("test")
except:
  nltk.download('wordnet')

try:
  test_tagger = nltk.pos_tag('test')
except:
  nltk.download('averaged_perceptron_tagger')

try:
  test_stop = stopwords.words("english")
except:
  nltk.download("stopwords")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.util import ngrams

try:
  import numpy as np
except:
#   !pip install numpy
  pass

try:
  from keras.models import load_model
except:
#   !pip install keras
  pass

import pickle
import json
import random
import re
# import pyaudio

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.ADV

def remove_repeated_words(text):
    """Removes consecutive repeated occurrences of words within a string, leaving only the first occurrence.
    
    text: (string)
    

Example:
    ---------
    >>> text = remove_repeated_words("I really really really like data science.")
    >>> print(text)
    I really like data science.
    
    """
    
    to_remove = []
    txt = text.split()
    for i in range(1, len(txt)):
        # Check whether a word is the same as the word before it, and if it is, append i to to_remove
        if txt[i] == txt[i-1]:
            to_remove.append(i)


    # Remove words by index (so remove in reverse order to keep the indices correct)
    to_remove.reverse()
    for i in to_remove:
        txt.pop(i)
    return ' '.join(t for t in txt)

def n_gram_split(text, n=2, string_mode=True, keep_original=True):
  """ Takes in a string and returns a list of n_grams."""
  the_grams = ngrams(text.split(), n)
  gram_list = []
  for each in the_grams:
    if keep_original:
      gram_list.append(each[0])
    gram_list.append(each[0]+'_'+each[1])
  if string_mode:
    new_text = ''
    for each in gram_list:
      new_text += each+' '
    return new_text
  else:
    return gram_list


lemmatizer = WordNetLemmatizer()
def clean_up_sentence(text):
    # Lower text
    text = text.lower()

    # Remove URLs
    url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    urls = [u[0] for u in re.findall(url_pattern, text)]
    text = re.sub(url_pattern, ' ', text)
    
    # Remove html tags
    text = re.sub('<[A-Za-z0-9 /]+>', ' ', text)
    text = re.sub('<[A-Za-z0-9\s=/"-_]+>', ' ', text)
    
    # Remove "&#34;" and any other HTML entities
    text = text.replace("&#\d*;", " ")
    
    # Remove non-alphanumeric characters
    text = re.sub("[\W+']", ' ', text)
    text = re.sub("_", ' ', text)

    # Remove @
    text = re.sub(r"@\S+", " ", text)

    # Remove http
    text = re.sub(r"http://\S+", " ", text)
    text = re.sub(r"https://\S+", " ", text)

    # Modify contractions
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)

    # Remove 
    text = re.sub('\W', ' ', text)

    # Remove co. abbr
    text = re.sub(r"co ", " ", text)

    # Remove extra spaces
    text = re.sub('\s+', ' ', text)

    # Strip off spaces
    text = text.strip(' ')

    ### Probably should split the logic here ###

    # Tokenize the text (split them into lists of words)
    text_tokens = nltk.word_tokenize(text)
    # text_tokens = text.split()
    # sw = stopwords.words("english")
    # text_tagged = [t for t in nltk.pos_tag(text_tokens) if t[0] not in sw]
    text_tagged = [t for t in nltk.pos_tag(text_tokens)]
    final_text = ' '.join(lemmatizer.lemmatize(w[0], pos=get_wordnet_pos(w[1])) for w in text_tagged)
    remove_repeated_words(final_text)
    # final_text = [lemmatizer.lemmatize(word.lower()) for word in text_tokens]

    return final_text

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('jsonIntents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = clean_up_sentence(pattern)
        w = n_gram_split(w, n=2, string_mode=False, keep_original=True)
        # w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('listWords.pkl','wb'))
pickle.dump(classes,open('listClasses.pkl','wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    # pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('language_model0.h5', hist)

print("model created")
