import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load model
with open('fishknn.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Fish Classification using KNN")
length = st.number_input("Length")
weight = st.number_input("Weight")
w_l_ratio = st.number_input("Width-to-Length Ratio")

# Predict button
if st.button("Predict"):
    prediction = model.predict([[length, weight, w_l_ratio]])
    st.write(f"Predicted Species: {prediction[0]}")
