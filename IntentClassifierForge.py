#!/bin/bash/python3

# Scripted installs for Colab use
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

# try:
#   test_stop = stopwords.words("english")
# except:
#   nltk.download("stopwords")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# from nltk.corpus import stopwords
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
  """Takes in a string and returns a list of n_grams or a modified string 
  containing of sequenced n_grams. With or without the original tokens.
  
    text: (string), 
    n: (number of tokens per gram), 
    string_mode: (True=returns string, False=returns tokens),
    keep_original: (True=return contains original tokens, False=return is contains only n_grams)
        

    Example:
        ---------
        >>> text = n_gram_split("I really really really like data science.")
        >>> print(text)
        I I_really really really_like like like_data data data_science science.
    """
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

    # Tokenize the text (split them into lists of words)
    text_tokens = nltk.word_tokenize(text)
    text_tagged = [t for t in nltk.pos_tag(text_tokens)]
    final_text = ' '.join(lemmatizer.lemmatize(w[0], pos=get_wordnet_pos(w[1])) for w in text_tagged)
    remove_repeated_words(final_text)
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

        # Tokenize each intent pattern
        w = clean_up_sentence(pattern)

        # Get n_gram tokens
        w = n_gram_split(w, n=2, string_mode=False, keep_original=True)
        
        # Add all grams to words list
        words.extend(w)
        
        # Add intents and gram patterns to documents list
        documents.append((w, intent['tag']))

        # Generate class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Clean up
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Output stats
print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)

# Pickle words and classes for use in production
pickle.dump(words,open('listWords.pkl','wb'))
pickle.dump(classes,open('listClasses.pkl','wb'))

# Generate training data
training = []
output_empty = [0] * len(classes)

# Generate 1 hot bag of words arrays for each intent pattern
for doc in documents:
    bag = []
    pattern_words = doc[0]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and split our dataset
random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Build a basic sequential model for categorization
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Set optimizer (SGD with Nesrov)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compile model with categorical loss
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model for 200 epochs
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save model for production
model.save('language_model0.h5', hist)

# Signal completion
print("Model Created")
