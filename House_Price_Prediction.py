import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("train.csv")
print("shape:",df.shape)
print("\nColumn:",df.columns.tolist())
print("\nData types:",df.dtypes)
print("Missing value:",df.isnull().sum()) # check missing value
print("\n First 5 Rows:")
print(df.head())

#-------------------------------------------------------------------------------------
sns.histplot(df["SalePrice"],kde=True)
#plt.show()

print(df.isnull().sum())

# Fill missing values with median -------------------------------------------------
df['LotFrontage'].fillna(df['LotFrontage'].median(),inplace=True)


# check categorical columns with missing values
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        print(f"{col}: {df[col].isnull().sum()} missing")
        # Fill with mode
      
        df[col].fillna(df[col].mode()[0], inplace=True)

# Verify no missing values----------------------------------------------------------
print("\nAfter filling:")
print(df.isnull().sum())


from sklearn.preprocessing import StandardScaler

#drop columns with 50% missing data -------------------------------------------------------
missing_precentage = (df.isnull().sum()/len(df)*100)
cols_to_drop = missing_precentage[missing_precentage > 50].index.tolist()
print(f"Dropping Columns : {cols_to_drop}")
df.drop(columns=cols_to_drop, inplace=True)

# Check remaining NaN values
print("Remaining NaN values:")
print(df.isnull().sum().sum())  # Total NaN count

# Drop columns that are 100% NaN
df = df.dropna(axis=1, how='all')

# For remaining columns, fill NaN:
# Numerical columns = fill with median
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Categorical columns = fill with mode
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Final check
print("\nAfter cleaning:")
print(f"Total NaN: {df.isnull().sum().sum()}")
print(f"Dataset shape: {df.shape}")



# one hotcoding for categorical columns --------------------------------------------------------
categorical_cols =df.select_dtypes(include=["object"]).columns.tolist()
print(f"categorical columns: {categorical_cols}")


df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"\nShape before encoding: {df.shape}")
print(f"Shape after encoding: {df_encoded.shape}")

# Separate features and target ---------------------------------------------------------------
x = df_encoded.drop('SalePrice', axis=1)
y = df_encoded['SalePrice']

print(f"\nFeatures (x) shape: {x.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nData ready for model training!")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

print(f"Training set :{x_train.shape}")
print(f"Testing set :{x_test.shape}")

#linear regression model
lr_model = LinearRegression()
lr_model.fit(x_train,y_train)
y_pred_lr = lr_model.predict(x_test)

# random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)

# Evaluate Both Models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)


    print(f"\n{model_name}:")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f" R² Score: {r2:.4f}")

evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Step 5: Compare predictions
print("\nComparison (First 5 houses):")
comparison_df = pd.DataFrame({
    'Actual': y_test.head().values,
    'LR_Prediction': y_pred_lr[:5],
    'RF_Prediction': y_pred_rf[:5]
})
print(comparison_df)




# Graph 1: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest: Actual vs Predicted')
plt.savefig('actual_vs_predicted.png')
#plt.show()

# Graph 2: Residuals (Error Distribution)
residuals = y_test - y_pred_rf
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals (Error)')
plt.title('Residual Plot')
plt.savefig('residuals.png')
#plt.show()

# Graph 3: Feature Importance
feature_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Top 15 Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
#plt.show()

print(feature_importance)

#---------------------------------------------------------------------------------------------
import joblib

# Save the trained model
joblib.dump(rf_model, 'house_price_model.pkl')
print(" Model saved: house_price_model.pkl")
