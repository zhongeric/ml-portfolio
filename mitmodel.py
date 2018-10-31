from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import shuffle
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import time
import pickle
from textblob import TextBlob

train = [
      ('Whats your favorite color', 'Baby Blue'),
      ('What do you do outside of school', 'I teach chess lessons and develop develop mobile and web apps')
]

test = [
       ('Color', 'Baby Blue'),
       ('Activities', 'I teach chess lessons and develop develop mobile and web apps')
]

from textblob.classifiers import NaiveBayesClassifier
with open('mitwebanswers.csv', 'r') as fp:
    cl = NaiveBayesClassifier(fp, format="csv")

pred = cl.prob_classify("Please tell me what your favorite color is")
print(pred.max())
print(pred.prob(pred.max()))

filename = 'mit_model.pkl'
pickle.dump(cl, open(filename, 'wb'))

print("successfuly saved model")
