# House Price Prediction Project

## About the project
This is a simple machine learning project where I tried to predict house prices using Linear Regression.  
The main goal was to understand how features like area, number of bedrooms and locality affect the price of a house.

---

## What I did
1. I created a small dataset with columns like area (in square feet), bedrooms, locality and price in lakhs.  
2. Then I converted the locality column into numeric form using dummy variables.  
3. I divided the data into training and testing parts using train_test_split.  
4. I trained a Linear Regression model on the training data.  
5. Finally, I checked the accuracy and tested some new values to see the predictions.

---

## Tools and Libraries used
- Python  
- Pandas  
- scikit-learn  
- Jupyter Notebook  

---

## Model performance
The model gives a good accuracy (R² score was around 0.8 depending on the data).  
It is working fine for basic price prediction.

---

## Example prediction
I tested with the following values:

| Area | Bedrooms | Locality | Predicted Price (Lakhs) |
|------|-----------|-----------|--------------------------|
| 1600 | 2 | Kollur | 85 |
| 1600 | 2 | Mankhal | 92 |

---

## How to run the notebook
1. Open the Jupyter Notebook file `house_price_prediction.ipynb`  
2. Run all the cells step by step  
3. You can change the input values in the last part to test your own predictions  


