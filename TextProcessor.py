#!/bin/bash/python3

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer

# import pickle
# import numpy as np
# from keras.models import load_model
# import json
# import random
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.util import ngrams


lemmatizer = WordNetLemmatizer()

class ProcessText():
    lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(self, treebank_tag):
        """Tags words as to part of speech"""
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

    def remove_repeated_words(self, text):
        """Removes consecutive repeated occurrences of words within a string"""

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

    def n_gram_split(self, text, n=2, string_mode=True, keep_original=True):
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

    def clean_up_sentence(self,text):
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
        final_text = ' '.join(lemmatizer.lemmatize(w[0], pos=self.get_wordnet_pos(w[1])) for w in text_tagged)
        self.remove_repeated_words(final_text)
        # final_text = [lemmatizer.lemmatize(word.lower()) for word in text_tokens]

        return final_text

