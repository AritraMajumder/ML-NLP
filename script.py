import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from util import *
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv")

df = df.fillna(" ")

#use title and author column as indep vars
df['content'] = df['author']+df['title']

#stemming done
df['content'] = df['content'].apply(stemming)

x = df['content'].values
y = df['label'].values


#using tfidf
vect = TfidfVectorizer()
vect.fit(x)

X = vect.transform(x)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)

#using logistic regression
model = LogisticRegression()
model.fit(x_train,y_train)


#training
test_pred = model.predict(x_test)
test_acc = accuracy_score(test_pred,y_test)

print('Accuracy: ',test_acc)


#single prediction
news = x_test[0]
print('Prediction: ',model.predict(news))
# 1 is fake 0 is real