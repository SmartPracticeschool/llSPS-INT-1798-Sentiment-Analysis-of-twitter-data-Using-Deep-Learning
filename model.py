

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('twitter data.csv')

import re
import nltk
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
data=[]

for i in range(0,31962):
        review=dataset["tweet"][i]
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        data.append(review)
        
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(data).toarray()
y=dataset.iloc[:,1].values

import pickle
pickle.dump(cv, open("cv.pkl", "wb"))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()



model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 1500))

model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train,y_train,batch_size=32,epochs=50)
model.save('mymodel.h5')