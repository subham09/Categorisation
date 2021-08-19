import logging

import pandas as pd

import numpy as np

from numpy import random

from imblearn.over_sampling import SMOTE

import nltk

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

import re
from bs4 import BeautifulSoup

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('Companies details v2.csv')

df = df[pd.notnull(df['name'])]

#print(df.head(10))

#print(df['long_desc'].apply(lambda x: len(x.split(' '))).sum())

my_names = ['health care', 'privacy and security', 'sports','agriculture and farming']


def print_plot(index):
    example = df[df.index == index][['long_desc', 'name']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Name:', example[1])

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):

    #text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = str(text)

    text = text.lower() # lowercase text

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text

    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text

    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text

    return text

df['long_desc'] = df['long_desc'].apply(clean_text)

#print_plot(10)

df['long_desc'].apply(lambda x: len(x.split(' '))).sum()

X = df.long_desc
y = df.name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
##sm = SMOTE(random_state=2)
##X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

nb = Pipeline([('vect', CountVectorizer()),

               ('tfidf', TfidfTransformer()),

               ('clf', MultinomialNB()),

              ])

nb.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred,target_names=my_names))


















