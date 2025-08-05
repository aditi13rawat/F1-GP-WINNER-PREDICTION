# 🏎️ F1 Grand Prix Winner Predictor

Predicts the winning probability of a Formula 1 driver based on:
- Starting Grid Position
- Driver
- Constructor
- Circuit

Built using:
- Machine Learning (Logistic Regression / RandomForest)
- Streamlit for the web interface
- Trained on historical F1 data

📊 Hosted App: [Click to Try](https://f1-gp-winner-prediction-djzt473yszdysp6ac85e8r.streamlit.app/)

![App Screenshot](./"C:\Users\Aditi Rawat\Downloads\Screenshot 2025-08-05 223955.png"/>)

## 🚀 Features
- Select driver, constructor, and circuit
- Input grid position
- Get real-time win probability prediction
- Responsive and minimal UI

## 🧠 Tech Stack
- Python
- Pandas, scikit-learn, joblib
- Streamlit
- F1 Dataset (Kaggle or Ergast API)

## 📂 Files in Repo
- `app.py` – Streamlit frontend
- `*.pkl` – Trained model and encoders
- `F1_RACE_GP_WINNER_PREDICTION_MODEL.ipynb` – Training notebook

## 📌 How to Run Locally
```bash
git clone https://github.com/aditi13rawat/F1-GP-WINNER-PREDICTION.git
cd F1-GP-WINNER-PREDICTION
pip install -r requirements.txt
streamlit run app.py
