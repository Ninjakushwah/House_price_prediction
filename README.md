# House Price Prediction Model

## Problem
Real estate agents estimate house values manually — slow and inconsistent. 
This model automates predictions with 89% accuracy.

## Dataset & Approach

**Data:**
- 1,460 houses from Kaggle
- 81 original features (structural, garage, basement, etc.)
- Split: 1,168 training, 292 testing (80/20)

**Process:**
1. Missing value handling: Median for numerical, mode for categorical
2. Feature encoding: One-hot encoded 23 categorical features → 155 total features
3. Model training: Random Forest (100 trees, default parameters)
4. Evaluation: Compared against Linear Regression baseline

## Results

**Random Forest Performance:**
- R² Score: 0.8952 (89.52%)
- Mean Absolute Error: $17,481
- RMSE: $28,352

**vs Baseline (Linear Regression):**
- R² Score: 0.6555 (65.55%)
- 24% improvement in R² by capturing non-linear patterns

## Why Random Forest?

Feature importance analysis showed Over all quality (54%) and GrLivArea (15%) dominate 
the prediction. This concentration indicates non-linear feature interactions — which 
Linear Regression cannot capture but Random Forest handles well.

## Model Limitation

Residual analysis revealed **heteroscedasticity**: 
- Cheap houses ($100-300K): ±$30K prediction error (tight)
- Expensive houses ($400K+): ±$150K prediction error (loose)

**Insight:** Luxury house pricing likely depends on external factors (location prestige, 
market trends) not captured in structural features alone. Model is reliable for typical 
properties but needs refinement for high-end market.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit

## Files
- 'House_Price_Prediction.py` — Model training & evaluation
- 'app.py` — Streamlit web interface
- 'house_price_model.pkl` — Trained Random Forest model
- 'train.csv` — Kaggle dataset

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open: http://localhost:8501

## Next Steps (If Improving)
- Stratified cross-validation for robust accuracy estimate
- Weighted loss function to penalize expensive house errors more
- Feature engineering for luxury segment 

---
Last updated: May 10, 2026
