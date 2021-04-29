import json
import csv
import numpy as np
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm
import random
import joblib
import time
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

# Getting Text Processing Tools

nltk.download('all')

# Importing Tools

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import RegexpTokenizer 

wordlist = words.words()

import spacy

stopword_set = set(stopwords.words('english'))
for idx in range(len(wordlist)):
    wordlist[idx] = wordlist[idx].lower()
wordlist = list(set(wordlist))

with open('train.json', 'r+') as f:
    records = json.load(f)

with open('test.json', 'r') as f:
    gold_test_list = json.load(f)

X_train = []
Y_train = []

X_test = []
Y_test = []

for item in records:
  X_train.append(item['content'])
  Y_train.append(item['label'])

for item in gold_test_list:
  X_test.append(item['content'])
  Y_test.append(item['label'])

def clean(s):
    # takes an input string
    # preprocesses it for the tf-idf vectorizer

    s.replace("\n", " ")
    tokens = word_tokenize(s)
    output = ""
    
    for token in tokens:
        unit = token.strip().lower()
        if unit in stopword_set or unit in punctuation:
            continue
        output = output + " " + unit
        
    return output.strip()

vectorizer = TfidfVectorizer(
        sublinear_tf = True,
        norm = "l2",
        encoding = 'utf-8',
        max_features = 512,
        stop_words = 'english',
        ngram_range = (1, 3),
        strip_accents = 'unicode',
        smooth_idf = True)

# To verify correctness of Vectorizer

X_train_vec = vectorizer.fit_transform(X_train)
print(np.shape(X_train_vec))

print("Size of Train: " + str(len(X_train)))
print("Size of Test: " + str(len(X_test)))
max_feature_size = 10000

def train(X, y, active = 'identity', solve = 'sgd', approach = 'mlp'):
    start = time.time()
    vec = vectorizer.fit(X)
    X_train_vec = vec.transform(X)
    
    if approach == 'lda':
        model = LinearDiscriminantAnalysis()
        model.fit(X_train_vec.toarray(), y)
    
    elif approach == 'mlp':
        model = MLPClassifier(alpha = 0,
                              hidden_layer_sizes = (512, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 1),
                              random_state = 2020,
                              activation = active,
                              max_iter = int(1e3),
                              solver = solve,
                              learning_rate = 'adaptive',
                              early_stopping = True,
                              momentum = 0.9,
                              batch_size = 512)
        
        model.fit(X_train_vec, y)
    
    end = time.time()
    time_to_train = int(round(end - start))

    hours = int(time_to_train / 3600)
    minutes = int(int(time_to_train % 3600) / 60)
    seconds = int(time_to_train % 60)

    print()
    print('Time taken for training: ' + str(hours).zfill(2) + ':' +
          str(minutes).zfill(2) + ':' + str(seconds).zfill(2))
    return vec, model

def get_res(vec, clf):
    X_test_vec = vec.transform(X_test)
    pred_Y_test = clf.predict(X_test_vec)
    print("Number of Features: " + str(np.shape(X_test_vec)[1]))
    print(classification_report(Y_test, pred_Y_test, digits = 6))
    return

# Best Setting for the tf-idf vectorizer based on the LDA Scheme
# sublinear_tf and smooth_idf set to True
# norm set to 'l2'

# To Try out all possibilities

try_all = False

if try_all == True:
    activations = ['identity', 'tanh', 'relu']
    solvers = ['adam', 'sgd', 'lbfgs']
else:
    activations = ['tanh']
    solvers = ['sgd']

for active in activations:
    for solver in solvers:
        if active == 'tanh' and solver == 'lbfgs':
            continue
        vec, model = train(X_train, Y_train, active, solver)
        print("Hidden Layer Activation = " + str(active) + ", Solver = " + str(solver))
        get_res(vec, model)
        
# Comments: ReLU does not perform well
# tanh activation with sgd solver gave the best results

# Testing out a basic pipeline

pipe = Pipeline([('Feature Builder', vec), ('Classifier', model)])
pred_Y_test = pipe.predict(X_test)
print(classification_report(Y_test, pred_Y_test, digits = 6))

# K-fold Cross Validation

X = X_train
Y = Y_train

def cross_val(algo = 'mlp', splits = 5):
    global X, Y
    splits = int(splits)
    if splits > 9 or splits < 3:
        splits = 5
    print("Classification Technique: " + str(algo))
    kf = KFold(n_splits = splits, shuffle = True, random_state = 2020)
    index = 1    

    for train_index, test_index in kf.split(X):
        X_train = []
        X_test = []
        Y_train = []
        Y_test = []

        for idx in train_index:
            X_train.append(X[idx])
            Y_train.append(Y[idx])

        for idx in test_index:
            X_test.append(X[idx])
            Y_test.append(Y[idx])

        if algo == 'lda':
            vec, model = train(X_train, Y_train, '', '', 'lda')
        else:
            vec, model = train(X_train, Y_train, 'tanh', 'sgd', 'mlp')

        pipe = Pipeline([('Feature Builder', vec), ('Classifier', model)])
        pred_Y_test = pipe.predict(X_test)

        print("Fold Index: " + str(index))
        index += 1
        print(classification_report(Y_test, pred_Y_test, digits = 6))
        
    return

# Performing K-Fold Cross Validation using LDA

cross_val('lda', splits = 3)

# Performing K-Fold Cross Validation using MLP

cross_val('mlp', splits = 3)

# Training a LDA Classifier on the complete dataset
# And saving the full pipeline into a Model

vec, model = train(X, Y, '', '', 'lda')

pipe = Pipeline([('Feature Builder', vec), ('Classifier', model)])
joblib.dump(pipe, "tf-idf_lda_model.pkl")

# Training a MLP Classifier on the complete dataset
# And saving the full pipeline into a Model

vec, model = train(X, Y, 'tanh', 'sgd', 'mlp')

pipe = Pipeline([('Feature Builder', vec), ('Classifier', model)])
joblib.dump(pipe, "tf-idf_mlp_model.pkl")

# Testing out the saved pipeline on all sample datapoints

saved_pipe = joblib.load("tf-idf_lda_model.pkl")

pred_Y_all = saved_pipe.predict(X)
print(classification_report(Y, pred_Y_all, digits = 6))

# Testing out Saved LDA Model on Test Data

saved_pipe = joblib.load("tf-idf_lda_model.pkl")

X_gold_test = []
Y_gold_test = []

for unit in gold_test_list:
    X_gold_test.append(unit['content'])
    Y_gold_test.append(unit['label'])
    
pred_Y_gold_test = saved_pipe.predict(X_gold_test)
print(classification_report(Y_gold_test, pred_Y_gold_test, digits = 6))

# Testing out Saved MLP Model on Test Data

saved_pipe = joblib.load("tf-idf_mlp_model.pkl")

X_gold_test = []
Y_gold_test = []

for unit in gold_test_list:
    X_gold_test.append(unit['content'])
    Y_gold_test.append(unit['label'])
    
pred_Y_gold_test = saved_pipe.predict(X_gold_test)
print(classification_report(Y_gold_test, pred_Y_gold_test, digits = 6))
