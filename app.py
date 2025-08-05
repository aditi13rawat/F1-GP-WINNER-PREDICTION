import streamlit as st
import joblib

# Load model and encoders
model = joblib.load("f1_winner_model.pkl")
le_driver = joblib.load("driver_encoder.pkl")
le_constructor = joblib.load("constructor_encoder.pkl")
le_circuit = joblib.load("circuit_encoder.pkl")

st.set_page_config(page_title="F1 GP Winner Predictor")

st.title("🏁 F1 Grand Prix Winner Predictor")
st.markdown("Predict a Formula 1 race winner based on qualifying data.")

# Input form
grid = st.number_input("Starting Grid Position", min_value=1, max_value=20, step=1)

driver = st.selectbox("Select Driver", le_driver.classes_.tolist())
constructor = st.selectbox("Select Constr
