import streamlit as st
import torch
from cpg_detector import CpGPredictor, preprocess_input

# Load the model
@st.cache_resource
def load_model():
    model = CpGPredictor()
    model.load_state_dict(torch.load("cpg_predictor_model.pth"))
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("CpG Predictor")
st.write("Enter a DNA sequence to predict the CpG count:")

# User input
user_input = st.text_input("DNA Sequence:", value="")

if user_input:
    if not all(char in "NACGT" for char in user_input.upper()):
        st.error("Invalid DNA sequence! Please enter a sequence containing only 'N', 'A', 'C', 'G', or 'T'.")
    else:
        # Preprocess the input
        input_tensor = preprocess_input(user_input.upper())

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()

        st.success(f"Predicted CpG count: {prediction:.2f}")
