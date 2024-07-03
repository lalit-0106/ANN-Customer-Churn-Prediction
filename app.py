import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoder and scalar
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')


## Background
def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://www.shutterstock.com/image-vector/incentive-welfare-program-employee-retention-600nw-1983348656.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
   

## Calling the Background function
set_bg_hack_url()

## User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number Of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_number = st.selectbox('Is Active Number', [0, 1])

# User input
input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    # 'Geography' : [label_encoder_gender.transform([gender])[0]],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card], 
    'IsActiveMember' : [is_active_number],
    'EstimatedSalary' : [estimated_salary]
})


## 'One hot' encode the geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


input_data_scaled = scaler.transform(input_data)


prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn probability is : {prediction_proba}')
if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')