# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:27:39 2024

@author: Srinivas
"""

#----------------------------------------------------------------------------
# 1. Import the required libraries
#----------------------------------------------------------------------------
import numpy as np
import pickle
import streamlit as st

#----------------------------------------------------------------------------
# 2. Loading the saved ML model
#----------------------------------------------------------------------------
model = pickle.load(open('FoodDesert_Classification_Iteration2_ML_Model_1.pkl', 'rb'))

#----------------------------------------------------------------------------
# 3. Creating a function for Prediction
#----------------------------------------------------------------------------
def fooddesert_prediction(input_data):    

    # changing the input_data to numpy array
    input_data = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data = input_data.reshape(1,-1)

    predict_label = model.predict(input_data)   # prediction
    
    if (predict_label[0] == 0):
        return 'Not a Food Desert'
    else :
        return 'Food Desert'

#--------------------------------------------------------------------------
# 4. Take Input from User
#--------------------------------------------------------------------------

# Titel of the App
    
st.title('Brooklyn Food Desert Area Prediction App - v1')
    
# getting the input data from user    
    
Medfamilyincome = st.text_input('Tract median family income')
SNAP = st.text_input('Proportion of households with public assistance income or food stamps')
PovertyRate = st.text_input('Poverty Rate')
TractSNAP = st.text_input('Total count of housing units receiving SNAP benefits in tract')
    
# prediction from 'fooddesert_prediction' funtion will be stored in this string variable 
Class_Prediction = ''  # empty string
    
# creating a button for Prediction   
#st.button(label="Food Desert Prediction Result", style="background-color: #DD3300; color:#eeffee; border-radius: 0.75rem;")
if st.button('Food Desert Prediction Result'):
    Class_Prediction = fooddesert_prediction([Medfamilyincome,SNAP,PovertyRate,TractSNAP])
        
    # Display a success message    
    st.success(Class_Prediction)
