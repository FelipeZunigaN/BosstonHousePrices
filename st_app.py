import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Model Prediction Test")

# Scaler and Model Loading
regmodel = pickle.load(open('regmodel.pkl','rb') )
scaler = pickle.load(open('scaling.pkl', 'rb'))

df = pd.read_csv('wine.csv')

if st.checkbox("Ver Tabla"):
    df

st.scatter_chart(data=df, x='alcohol', y='color_intensity')

CRIM = st.number_input("CRIM :", value=0.006)
ZN = st.number_input("ZN :", value=18)
INDUS = st.number_input("INDUS :", value=2.31)
CHAS = st.number_input("CHAS :", value=0)
NOX = st.number_input("NOX :", value=0.538)
RM = st.number_input("RM :",value= 6.575)
AGE = st.number_input("AGE :", value=65.2)
DIS = st.number_input("DIS :", value=4.0900)
RAD = st.number_input("RAD :", value=1.0)
TAX = st.number_input("TAX :", value=296.0)
PTRATIO = st.number_input("PTRATIO :", value=15.3)
B = st.number_input("B :", value=396.90)
LSTAT = st.number_input("LSTAT :", value=4.98)

cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

def predict():
    values = np.array([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]).reshape(1,13)
    scaled_inputs = scaler.transform(values)
    prediction = regmodel.predict(scaled_inputs)[0]

    st.success(f"Predicted Value: {prediction}")

    # return st.write(f"Predicted Value: {prediction}")

st.button('Predict', on_click=predict)