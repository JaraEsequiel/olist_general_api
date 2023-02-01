import re
import string
import time
import unicodedata
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

InteractiveShell.ast_node_interactivity =  'all'
nltk.download('stopwords')
nltk.download('punkt')

from deep_translator import GoogleTranslator
from textblob import TextBlob


# Transforming the reviews data by removing stopwords, using regular expressions module to accept only letters, 
# making all the words lower case for consistency and joining them into a 'comments' list.
# We'll call this function from tokenize_comments

def preprocess_comments(text):
    text = str(text)
    comments = []
    stop_words = set(stopwords.words('portuguese'))
    
    only_letters = re.sub(r'[^a-zA-ZÀ-ÿ]', " ", text)
    lower_case = only_letters.lower()
    filtered_result = ' '.join([l for l in lower_case.split() if l not in stop_words]) 
    comments.append(filtered_result)

    return comments     


# Tokenizing comments to make easier their preprocessing

def tokenize_comments(comment):
    comment = str(comment)    

    tokens = word_tokenize(comment, language= 'portuguese')
    clean_tokens = preprocess_comments(tokens)
    while("" in clean_tokens):
        clean_tokens.remove("")
    clean_tokens = ' '.join([str(elem) for elem in clean_tokens])
  
    return clean_tokens

# English translation for TextBlob

def translate(text):
    text = str(text)
    translated = GoogleTranslator(source='auto', target='en').translate(text)
    return translated

# Polarity is the measure of the overall combination of the positive and negative emotions in a sentence.
# For TextBlob, Polarity is the output that lies between [-1,1], where -1 refers to negative sentiment and +1 refers to positive sentiment

def getPolarity(text):
    text = str(text)
    analysis = TextBlob(text)
    if not pd.isna(text): 
        if text == 'boa' or text =='bom' or text =='recomendo':
            result = 0.4
            return result
        elif text != '':    
            result = analysis.sentiment.polarity
            return result           
    else: return 100

# We measure and assign the sentiment class using the polarity value

def getSentimentClass(polarity):   
    try: 
        if polarity >= 0.3 and polarity <=1:
            return 'positive'
        elif polarity >= 0 and polarity < 0.3:
            return 'neutral'
        elif polarity < 0:
            return 'negative'        
        else: return 0
    except: return 'neutral'


# If both of the two options available for NLP in this dataset (review title and review message) are NaN values,
# we will use the reviewScore to get the sentiment class

def reviewScore (score):
    if score >= 4:
        result = 'positive'
        return result
    elif score >= 3:
        result = 'neutral'
        return result
    elif score <= 2:
        result = 'negative'
        return result


def getSentimentAnalysis (input):
    if input != np.NaN:
        if input != '':
            token = tokenize_comments(input)
            trans = translate(token)
            polar = getPolarity(trans)
            clas = getSentimentClass(polar)
            return clas
    
    return 'Error'