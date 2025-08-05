import streamlit as st
import joblib

# Inject custom background CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://www.hdcarwallpapers.com/walls/honda_f1_racing_car-wide.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .css-18e3th9 {{
        background-color: rgba(255, 255, 255, 0.9); 
        border-radius: 10px;
        padding: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load model and encoders
model = joblib.load("f1_winner_model.pkl")
le_driver = joblib.load("driver_encoder.pkl")
le_constructor = joblib.load("constructor_encoder.pkl")
le_circuit = joblib.load("circuit_encoder.pkl")

# Page config
st.set_page_config(page_title="F1 GP Winner Predictor", layout="centered")

# App Title
st.title("🏁 F1 Grand Prix Winner Predictor")
st.markdown("Predict the win probability of a Formula 1 driver based on qualifying data.")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    grid = st.number_input("Grid Position", min_value=1, max_value=20, step=1)
    driver = st.selectbox("Select Driver", le_driver.classes_.tolist())

with col2:
    constructor = st.selectbox("Select Constructor", le_constructor.classes_.tolist())
    circuit = st.selectbox("Select Circuit", le_circuit.classes_.tolist())

# Optional: Weather input (if added to model later)
# weather = st.radio("Track Weather", ["Dry", "Rainy"])
# weather_bin = 1 if weather == "Rainy" else 0

# Encode inputs
driver_enc = le_driver.transform([driver])[0]
constructor_enc = le_constructor.transform([constructor])[0]
circuit_enc = le_circuit.transform([circuit])[0]

# Prepare input
input_data = [[grid, driver_enc, constructor_enc, circuit_enc]]  # Add weather_bin or dnf_rate if needed

# Predict button
if st.button("Predict Win Probability"):
    prob = model.predict_proba(input_data)[0][1]
    st.success(f"🏆 Predicted Win Probability: {prob:.2%}")
