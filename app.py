
from keras.models import load_model
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
global graph

graph = tf.compat.v1.get_default_graph()
cv = pickle.load(open('cv.pkl', 'rb'))


model=load_model('mymodel.h5')
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/y_predict',methods=['POST'])
def y_predict():
    """
        For rendering results on HTML GUI
    """
    model.predict(cv.transform(['I\'m So Happy']))
    x_test=[request.form['Sentence']]
    print('Positive')
    return render_template('index.html',prediction_text='Positive')
    x_test = cv.transform([request.form['Sentence']])
    return render_template('index.html',prediction_text='Positive')
                           
    # with open('cv.pkl','rb') as file:
    #     cv=pickle.load(file)    
    #     model=load_model('mymodel.h5')
    #     prediction = model.predict(cv.transform([request.form['Sentence']]))
    #     output=prediction[0]
    #     if(output<0.5):
    #         return render_template('index.html',prediction_text="does not contain racial content..Positive"+str(output))
    #     else:
    #         return render_template('index.html',prediction_text="contains racial content..Negative"+str(output))
if(__name__=="__main__"):
    app.run(debug=True)