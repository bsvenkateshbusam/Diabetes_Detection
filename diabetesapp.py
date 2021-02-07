#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pickle
import streamlit as st


# In[3]:


lrmodel = open('lrdiabetes.pkl','rb')
classifier = pickle.load(lrmodel)


# In[4]:


def diabetes(Pregnancies,Glucose,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    values = np.array([[Pregnancies,Glucose,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]).astype(np.float64)
    prediction = classifier.predict(values)
    return prediction


# In[5]:


def main():
    st.title("Diabetics Prediction")
    
    Pregnancies = st.text_input('Pregnancies','')
    Glucose = st.text_input('Glucose','')
    BP = st.text_input('BP','')
    SkinThickness = st.text_input('SkinThickness','')
    Insulin = st.text_input('Insulin','')
    BMI = st.text_input('BMI','')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction','')
    Age = st.text_input('Age','')
    
    result =''
    
    if st.button("Predict"):
        result = diabetes(Pregnancies,Glucose,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    st.success(result)
    
    
if __name__ == '__main__':
    main()


# In[ ]:




