import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Classifier")

sepal_length = st.number_input("Sepal Length (cm)")
sepal_width = st.number_input("Sepal Width (cm)")
petal_length = st.number_input("Petal Length (cm)")
petal_width = st.number_input("Petal Width (cm)")

if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"Predicted Iris Species: {prediction[0]}")
