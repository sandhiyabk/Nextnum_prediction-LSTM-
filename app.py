import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load model
model = load_model("my_model.keras")

st.title("🔢 RNN Next Number Predictor")
st.write("Enter 3 consecutive numbers to predict the next one:")

a = st.number_input("First Number", value=98)
b = st.number_input("Second Number", value=99)
c = st.number_input("Third Number", value=100)

if st.button("Predict"):
    input_seq = np.array([[a, b, c]]).reshape(1, 3, 1)
    prediction = model.predict(input_seq, verbose=0)
    st.success(f"Predicted Next Number: {prediction[0][0]:.2f}")
