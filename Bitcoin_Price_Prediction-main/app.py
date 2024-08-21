import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def main():
    st.markdown("""
    <style>
        .header {
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #f0f0f0;
            padding: 10px;
            border: 2px solid #000;
            border-radius: 10px;
        }
        .header h1 {
            font-size: 24px;
            margin: 0;
        }
        .input-container {
            margin-bottom: 10px;
        }
        .input-container label {
            font-weight: bold;
        }
        .input-container input {
            width: 100%;
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='header'><h1>Stock Price Predictor</h1></div>", unsafe_allow_html=True)

    # Get user input for 'Open'
    open_val = st.text_input("Enter Open:", value="37599.410156", key="open")

    # Get user input for 'High'
    high = st.text_input("Enter High:", value="39478.953125", key="high")

    # Get user input for 'Low'
    low = st.text_input("Enter Low:", value="37243.972656", key="low")

    # Get user input for 'Volume'
    volume = st.text_input("Enter Volume:", value="35460750427", key="volume")

    if open_val and high and low and volume:
        # Load the model
        the_best_model = joblib.load('svr_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Prepare user input features
        user_input_features = np.array([float(open_val), float(high), float(low), float(volume)]).reshape(1, -1)

        # Scale user input features
        user_input_features = scaler.transform(user_input_features)

        # Make a prediction
        predicted_close = the_best_model.predict(user_input_features)

        # Display the prediction
        st.success(f"The predicted close price is: {predicted_close[0]:.2f}")

if __name__ == "__main__":
    main()
