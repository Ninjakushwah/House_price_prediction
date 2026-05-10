# House Price Prediction Model

## Problem
Real estate agents manually estimate house values, which is slow and inconsistent. 
This model automates the process with 89% accuracy.

## What I Did

**Dataset:**
- 1,460 houses from Kaggle
- 81 features (house size, bedrooms, quality, etc.)
- Split: 80% training (1,168), 20% testing (292)

**My Steps:**
1. Cleaned missing data (filled numeric with median, categories with mode)
2. Converted text categories to numbers using one-hot encoding
3. Trained Random Forest and compared with Linear Regression
4. Analyzed which features matter most and where predictions fail

## Results

**Random Forest Model:**
- Accuracy (R²): 0.8952 = predicts 89.5% of price variation correctly
- Average error: $17,481
- Error spread: $28,352 (RMSE)

**Why better than Linear Regression?**
- Linear Regression only got 65.5% accuracy
- Random Forest found hidden patterns that Linear Regression missed
- Top feature: OverallQuality (54% importance) — shows it's not just size, quality matters

## Important: Where It Fails

My model is really good for normal houses ($100K-300K) — errors are small (±$30K).

But for expensive houses ($400K+), errors get big (±$150K). 

Why? Because expensive house prices depend on things I don't have data for:
- Location prestige (neighborhood name, not just coordinates)
- Market trends at that time
- Unique property history

**So:** Use this model for typical houses. For luxury properties, it's less reliable.

## How to Use

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open: http://localhost:8501

## What I Learned

1. How to clean messy data
2. How to compare different models fairly
3. How to read model output (feature importance, residuals)
4. That 89% accuracy doesn't mean the model is perfect everywhere

## If I Had More Time

- Test on different data chunks (cross-validation) to make sure 89% is real
- Give more penalty to expensive house mistakes
- Create special features for luxury properties
