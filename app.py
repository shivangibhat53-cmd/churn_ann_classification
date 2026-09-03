import streamlit as st
import numpy as np 
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


#Load the trained model 
model = tf.keras.models.load_model('model.h5', compile=False)

##Load the trained model, scaler and ohe
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo  = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer Churn PRediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data

# 1. Create a raw input DataFrame matching your training features exactly
raw_input_df = pd.DataFrame({
    'Geography': [geography],
    'Gender': [gender]
})

# 2. Convert raw input data to the format your model expects
# Wrap the geography input in a DataFrame to satisfy the feature names requirement
geo_df = pd.DataFrame({'Geography': [geography]})
geo_encoded = onehot_encoder_geo.transform(geo_df)
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Encode gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# 3. Reconstruct your final DataFrame features
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Combine everything
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# CRITICAL CHECK: Ensure your column order perfectly matches training data
# If scaler still warns you, enforce the exact training columns order here:
# training_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France', 'Geography_Germany', 'Geography_Spain']
# input_data = input_data[training_columns]

# Scale using the DataFrame so it retains the feature names
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
