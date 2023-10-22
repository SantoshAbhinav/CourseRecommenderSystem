from re import I
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from scipy import hstack
from scipy import sparse
import random
import pickle
import streamlit as st
import nltk
import json
import string
from numpy import dot
from numpy.linalg import norm
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import wordpunct_tokenize
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from nltk.corpus import stopwords


df_edx = pd.read_pickle('pickle1')
st.write(df_edx)
title_vectorizer = CountVectorizer()
institution_vectorizer = CountVectorizer()
level_vectorizer = CountVectorizer()
title_features = title_vectorizer.fit_transform(df_edx['title'])
institution_features = institution_vectorizer.fit_transform(df_edx['institution'])
level_features = level_vectorizer.fit_transform(df_edx['Level'])
w1 = 1
w2 = 1
w3 = 1
st.write(title_features.shape)
df_features = np.concatenate([w1*title_features.todense(), w2*institution_features.todense(), w3*level_features.todense()], axis=1)
#df_features = sparse.csr_matrix(df_features)


def features(title, institution, level):
    filtered_text = " ".join(i for i in [j.lower() for j in wordpunct_tokenize(title) if j.lower() not in stopwords.words('english')]\
                             if ((i.isalpha()) and i not in string.punctuation) or i == " ")
    filtered_text = " ".join(j for j in filtered_text.split() if j != "introduction" and j != "de"\
                             and j != "introducción" and j != "en" and j != "part" and j != "fundamentals") 
    title_feature = title_vectorizer.transform([filtered_text])
    institution_feature = institution_vectorizer.transform(["".join(i for i in institution.split())])
    level_feature = level_vectorizer.transform([level])
    doc_feature = np.concatenate((w1*title_feature.todense(), w2*institution_feature.todense(), w3*level_feature.todense()), axis=1)
    st.write(doc_feature.shape)
    return doc_feature

def bag_of_words_model(title, institution, level):
    global df_features
    doc_feature = features(title, institution, level)
    # doc_feature = np.array(doc_feature)
    # df_features = np.array(df_features)
    st.write(df_features.shape)
    # pairwise_dist = []
    # for i in range(len(df_features)):
    #     pairwise_dist.append(dot(df_features[i], doc_feature)/(norm(doc_feature)*norm(df_features[i])))
    pairwise_dist = cosine_similarity(df_features, doc_feature)
    indeces = np.argsort(pairwise_dist.flatten())[-10:]
    pdists  = np.sort(pairwise_dist.flatten())[-10]
    #indeces = list(indeces)
    #indeces.sort(key = lambda x : len(df_edx['title'].iloc[x]))
    df = pd.read_csv("edX.csv")
    return df.iloc[indeces]


def reply(inp):
    stemmer = LancasterStemmer()
    k = " ".join(stemmer.stem(w) for w in wordpunct_tokenize(inp) if w.isalpha())
    t = ques_vectorizer.transform([k]).toarray()
    new_model = tf.keras.models.load_model('./model')
    y = new_model.predict([t])
    ind = y.argmax()
    out = random.choice(bot_data['intents'][ind]['responses'])
    return out


with open("intents.json") as file:
    bot_data = json.load(file)


ques = np.load("ques.npy")
ques_vectorizer = CountVectorizer()
ques_features = ques_vectorizer.fit_transform(ques).toarray()

st.title("EdX course recommendation system")
st.header("Chat with me!!!")
greeting = st.text_input("Chat with mini_bot")
if greeting:
    st.write(reply(greeting))
    #st.subheader("Enter the course/domain you are looking for :)")
    course = st.text_input("Enter the course/domain you are looking for :)")
    if course:
        #st.subheader("Enter the institution/univesity you are looking for :)")
        institution = st.text_input("Enter the institution/univesity you are looking for :)")
        if institution:
            #st.subheader("Enter the courses level :)")
            level = st.text_input("Enter the courses level :)")
            if level:
                st.dataframe(bag_of_words_model(course, institution, level))
                r = st.text_input("Anything missed!!")
                if r:
                    st.write(reply(r))