# House Price Prediction Model

A machine learning project that predicts house prices.

## What Does This Do?

This model looks at house features and predicts how much the house will cost.

**Example:**
- Input: House size = 2000 sq ft, Bedrooms = 3, Quality = Good
- Output: Predicted Price = 250,000

## Model Accuracy

- R² Score: 0.8952** ✓ (89.52% accurate)
- Average Error: 17,481
- Better than: Linear Regression (65% accurate)

## Dataset

- 1,460 houses
- 81 house features
- Source: Kaggle

## How I Built It

1. Data Cleaning - Fixed missing values
2. Feature Engineering - Converted text to numbers
3. Model Training - Used Random Forest
4. Comparison - Random Forest better than Linear Regression

## Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit

## Files

- "House_Price_prediction.py" - Training code
- "app.py" - Web app
- "house_price_model.pkl" - Saved model
- "actual_vs_predicted.png" - Performance graph
- "residuals.png" - Error graph

## How to Use

"""bash
pip install -r requirements.txt
streamlit run app.py
"""

Open: http://localhost:8501

## Results

**Random Forest:**
- R² = 0.8952 (89.52% accurate)
- MAE = 17,481
- RMSE = 28,352

**vs Linear Regression:**
- R² = 0.6555 (65.55%)
- MAE = 20,232
- RMSE = 51,405

## Skills

 Data preprocessing  
 Feature engineering  
 Model comparison  
 Model evaluation (R², MAE, RMSE)  
 Web app development (Streamlit)  
 Python, Pandas, Scikit-learn  

 
**Last Update**:08 May 2026
