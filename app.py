import streamlit as st
import joblib

# Load model and encoders
model = joblib.load("f1_winner_model.pkl")
le_driver = joblib.load("driver_encoder.pkl")
le_constructor = joblib.load("constructor_encoder.pkl")
le_circuit = joblib.load("circuit_encoder.pkl")

# Streamlit UI setup
st.set_page_config(page_title="F1 GP Winner Predictor", layout="centered")
st.title("🏁 F1 Grand Prix Winner Predictor")
st.markdown("Predict the win probability of a Formula 1 driver based on qualifying data.")

# User Inputs
grid = st.number_input("Starting Grid Position", min_value=1, max_value=20, step=1)

driver = st.selectbox("Select Driver", le_driver.classes_.tolist())
constructor = st.selectbox("Select Constructor", le_constructor.classes_.tolist())
circuit = st.selectbox("Select Circuit", le_circuit.classes_.tolist())

# Encode inputs
driver_enc = le_driver.transform([driver])[0]
constructor_enc = le_constructor.transform([constructor])[0]
circuit_enc = le_circuit.transform([circuit])[0]

input_data = [[grid, driver_enc, constructor_enc, circuit_enc]]

# Predict button
if st.button("Predict Win Probability"):
    prob = model.predict_proba(input_data)[0][1]
    st.success(f"🏆 Predicted Win Probability: {prob:.2%}")

