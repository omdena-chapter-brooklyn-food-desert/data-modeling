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
model = pickle.load(open('FoodDesert_Classification_StdzdData_Clusters_v4.pkl', 'rb'))

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
        return 'This Brooklyn area belongs to Class 0'
    elif (predict_label[0] == 1):
        return 'This Brooklyn area belongs to Class 1'
    elif (predict_label[0] == 2):
        return 'This Brooklyn area belongs to Class 2'
    elif (predict_label[0] == 3):
        return 'This Brooklyn area belongs to Class 3'
    else :
        return 'This Brooklyn area belongs to Class 4'

#--------------------------------------------------------------------------
# 4. Take Input from User
#--------------------------------------------------------------------------

# Titel of the App
    
st.title('Brooklyn Food Desert Area Prediction App - v1')
    
# getting the input data from user    
    
Medfamilyincome = st.text_input('Tract median family income')
SNAP = st.text_input('Proportion of households with public assistance income or food stamps')
PovertyRate = st.text_input('Poverty Rate')
TractKids = st.text_input('Total count of children age 0-17 in tract')
TractSeniors = st.text_input('Total count of seniors age 65+ in tract')
TractWhite = st.text_input('Total count of White population in tract')
TractBlack = st.text_input('Total count of Black or African American population in tract')    
TractAsian = st.text_input('Total count of Asian population in tract')
TractNHOPI = st.text_input('Total count of Native Hawaiian and Other Pacific Islander population in tract')
TractAIAN = st.text_input('Total count of American Indian and Alaska Native population in tract')
TractOMultir = st.text_input('Total count of Other/Multiple race population in tract')
TractHUNV = st.text_input('Total count of housing units without a vehicle in tract')
TractSNAP = st.text_input('Total count of housing units receiving SNAP benefits in tract')
    
# prediction from 'fooddesert_prediction' funtion will be stored in this string variable 
Class_Prediction = ''  # empty string
    
# creating a button for Prediction   
#st.button(label="Food Desert Prediction Result", style="background-color: #DD3300; color:#eeffee; border-radius: 0.75rem;")
if st.button('Food Desert Prediction Result'):
    Class_Prediction = fooddesert_prediction([Medfamilyincome,SNAP,PovertyRate,TractKids,TractSeniors,TractWhite,TractBlack,TractAsian,TractNHOPI,TractAIAN,TractOMultir,TractHUNV,TractSNAP])
        
    # Display a success message    
    st.success(Class_Prediction)
