#!/bin/bash/python3

import json, random, pickle
import numpy as np
from keras.models import load_model
from TextProcessor import ProcessText

class IntentModel():
    # load contents
    model = load_model('language_model0.h5')
    intents = json.loads(open('jsonIntents.json').read())
    words = pickle.load(open('listWords.pkl','rb'))
    classes = pickle.load(open('listClasses.pkl','rb'))

    # Create Processing object
    proctext = ProcessText()

    # Define bag of words
    def bow(self, sentence, words):

        # Tokenize the sentence
        sentence = self.proctext.clean_up_sentence(sentence)
        # Get N_grams
        sentence_words = self.proctext.n_gram_split(sentence, n=2, string_mode=False, keep_original=True)

        # Create bow array 
        bag = [0]*len(words)
        firstbag = bag
        for s in sentence_words:
            # print(s)
            for i,w in enumerate(words):
                # print(w)
                if w == s:
                    # Assign bow values 1 = present, 0 = not present
                    bag[i] = 1
        return(np.array(bag))

    def predict_class(self, sentence, model):

        # Get the BOW
        p = self.bow(sentence, self.words)

        # Run model inference
        res = model.predict(np.array([p]))[0]

        # Set error threshold
        ERROR_THRESHOLD = 0.25

        # Enumerate results above threshold
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

        # Sort and return results
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, ints, intents_json):

        # If we get results
        if ints != []:

            # Get prediction tag
            tag = ints[0]['intent']

            # Pick response from tagged response space
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if(i['tag']== tag):
                    result = random.choice(i['responses'])
                    break
        # If no results reply
        else: 
            result = "I'm sorry I didn't understand that!"
        return result