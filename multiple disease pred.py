# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:12:07 2023

@author: harsh
"""

import pickle
from sklearn.preprocessing import StandardScaler 
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

scaler1=StandardScaler()
diabetes=pd.read_csv('C:/Users/harsh/OneDrive/Desktop/Multiple disease prediction system/dia.csv', sep=',',header=0)
X = diabetes.drop(columns = 'diabetes', axis=1)
scaler1.fit(X)
scaler2=StandardScaler()
parkinsons=pd.read_csv('C:/Users/harsh/OneDrive/Desktop/Multiple disease prediction system/dataset/parkinsons.csv',sep=',',header=0)
Y=parkinsons.drop(columns='status',axis=1)
scaler2.fit(Y)


#loading the saved models
diabetes_model = pickle.load(open('C:/Users/harsh/OneDrive/Desktop/Multiple disease prediction system/saved models/diabetes_model.sav','rb'))
heart_disease_model = pickle.load(open('C:/Users/harsh/OneDrive/Desktop/Multiple disease prediction system/saved models/heart_disease_model.sav','rb'))
parkinsons_model = pickle.load(open('C:/Users/harsh/OneDrive/Desktop/Multiple disease prediction system/saved models/parkinsons_model.sav','rb'))

#sidebar for navigation

with st.sidebar:
    
    selected  = option_menu('Multiple Disease Prediction System',
                            ['Diabetes Prediction','Heart Disease Prediction','Parkinsons Prediction'],
                            icons = ['activity','heart','person'],
                            default_index = 0)
    
    
# Diabetes prediction page

if (selected == 'Diabetes Prediction'):
    
    #page title
    st.title('Diabetes Prediction ')
    
    #getting the input data from the user
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        Gender = st.selectbox('Gender',('M','F'))
        
        
    with col2:
        Age = st.text_input('Age of the person')
        
        
    with col3:
        Hypertension = st.selectbox('Hypertension',('Y','N'))
        
        
    with col1:
        Heartdisease =st.selectbox('Heartdisease',('Y','N'))
        
        
    with col2:
        BMI = st.text_input('BMI')
        
        
    with col3:
        HbA1c_level = st.text_input('HbA1c_level')
        
        
    with col1:
        bloodglucose = st.text_input('Glucose level')
    
    #code for prediction
    
    diab_dignosis = ''
    
    #creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        if not all([Gender,Age,Hypertension,Heartdisease, BMI,HbA1c_level,bloodglucose]):
            st.warning("Please fill in all the fields.")
            
        else:
        # changing the input_data to numpy array
            try:
                if(Gender=='M'):
                    Gender=1 
                else:
                    Gender=0
                if(Hypertension=='Y'):
                   Hypertension=1 
                else:
                    Hypertension=0
                if(Heartdisease=='Y'):
                   Heartdisease=1 
                else:
                    Heartdisease=0
            
                input_data=[Gender,Age,Hypertension,Heartdisease, BMI,HbA1c_level,bloodglucose]
                input_data_as_numpy_array = np.asarray(input_data)
        
                # reshape the array as we are predicting for one instance
                input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
                std_data = scaler1.transform(input_data_reshaped)
        
                diab_prediction = diabetes_model.predict(std_data)
                
                if (diab_prediction[0]==1):
                    diab_dignosis = 'The person is Diabetic'
                else:
                    diab_dignosis = 'The person is Not Diabetic'
            except(Exception ):
                st.warning("Invalid Entries")
                
        st.success(diab_dignosis)

 #Heart disease prediction
      
if (selected == 'Heart Disease Prediction'):
      
      #page title
     st.title('Heart Disease Prediction')
      
      #getting the input data from the user
      #columns for input fields
     col1, col2, col3 = st.columns(3)
     with col1:
         age = st.number_input('Age of the person')
          
          
     with col2:
         sex = st.selectbox('Sex', ('M','F'))
          
          
     with col3:
         cp = st.number_input('Chest Pain Types')
          
          
     with col1:
         trestbps = st.number_input('Resting Blood Pressure')
          
          
     with col2:
         chol = st.number_input('Serum cholestorol in mg/dl')
          
          
     with col3:
         fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
          
          
     with col1:
         restecg = st.number_input('Resting Electrocardiographic results')
          
          
     with col2:
         thalach = st.number_input('Maximum heart rate achieved')
          
          
     with col3:
         exang = st.number_input('Exercise induced angina')
          
          
     with col1:
         oldpeak = st.number_input('ST depression induced by exercise relative to rest')
          
          
     with col2:
         slope = st.number_input('Slope of the peak exercise ST segment')
          
          
     with col3:
         ca = st.number_input('Number of major vessels colored by flourosopy')
          
          
     with col1:
         thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
          
          
      
      #code for prediction
      
     heart_dignosis = ''
      
      #creating a button for prediction
      
     if st.button('Heart Disease Test Result'):
         try:
             if(sex=='M'):
                    sex=1
             else:
                    sex=0
             heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal  ]])
                  
             if (heart_prediction[0]==1):
                 heart_dignosis = 'The person is having heart disease'
             else:
                 heart_dignosis = 'The person does not have any heart disease'
         except(Exception ):
              st.warning("Invalid")
     st.success(heart_dignosis)

#Parkinsons Disease
if (selected == 'Parkinsons Prediction'):
    
    #page title
    st.title('Parkinsons Prediction')
    
    
    col1, col2, col3, col4, col5 = st.columns(5)  
   
    with col1:
        fo = st.text_input('MDVP: Fo(Hz)')
       
    with col2:
        fhi = st.text_input('MDVP: Fhi(Hz)')
       
    with col3:
        flo = st.text_input('MDVP: Flo(Hz)')
       
    with col4:
        Jitter_percent = st.text_input('MDVP: Jitter(%)')
       
    with col5:
        Jitter_Abs = st.text_input('MDVP: Jitter(Abs)')
       
    with col1:
        RAP = st.text_input('MDVP: RAP')
       
    with col2:
        PPQ = st.text_input('MDVP: PPQ')
       
    with col3:
        DDP = st.text_input('Jitter: DDP')
       
    with col4:
        Shimmer = st.text_input('MDVP: Shimmer')
       
    with col5:
        Shimmer_dB = st.text_input('MDVP: Shimmer(dB)')
       
    with col1:
        APQ3 = st.text_input('Shimmer: APQ3')
       
    with col2:
        APQ5 = st.text_input('Shimmer: APQ5')
       
    with col3:
        APQ = st.text_input('MDVP: APQ')
       
    with col4:
        DDA = st.text_input('Shimmer: DDA')
       
    with col5:
        NHR = st.text_input('NHR')
       
    with col1:
        HNR = st.text_input('HNR')
       
    with col2:
        RPDE = st.text_input('RPDE')
       
    with col3:
        DFA = st.text_input('DFA')
       
    with col4:
        spread1 = st.text_input('spread1')
       
    with col5:
        spread2 = st.text_input('spread2')
       
    with col1:
        D2 = st.text_input('D2')
       
    with col2:
        PPE = st.text_input('PPE')
       
   
   
   # code for Prediction
    parkinsons_diagnosis = ''
   # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        if not all([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]):
            st.warning("Please fill in all the fields")
        else:
            try:
                input_data2=[[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]]
                input_data_as_numpy_array2= np.asarray(input_data2)
        
                # reshape the array as we are predicting for one instance
                input_data_reshaped2 = input_data_as_numpy_array2.reshape(1,-1)
                std_data = scaler2.transform(input_data_reshaped2)
                parkinsons_prediction = parkinsons_model.predict(std_data)
            except(Exception ):
                    st.warning("Invalid")                       
            if (parkinsons_prediction[0] == 1):
              parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
              parkinsons_diagnosis = "The person does not have Parkinson's disease"
       
    st.success(parkinsons_diagnosis)

    
    

    