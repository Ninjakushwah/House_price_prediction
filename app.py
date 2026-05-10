import streamlit as st
import pandas as pd
import numpy as np


# page config
st.set_page_config(page_title="House Price Predictor", layout="wide")

# title

st.title("House Price Prediction")
st.write("Enter the details of the house to predict its price:")



# displya image

st.header("Model performance visualization")

col1,col2 =st.columns(2)

with col1:
    try:
      st.image("actual_vs_predicted.png", caption="Actual vs Predicted")

    except:
        st.warning("residuals.png not found")


# Model Info
st.header("Model Information")
st.write("""
- Model Type:-- Random forest regressor
- Training Samples:--1,168 houses
- Testing Samples:-- 292 houses
- R² Score:*-- 0.8952 (89.52%)
- Mean Absolute Error:-- $17,481.78
- RMSE:-- $28,352.11
""")

st.info("This is a demonstration of house price prediction using machine learning!")        




