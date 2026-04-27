# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load dataset, clean column names, and separate features (Size, Bedrooms) and targets (Price, Occupants).
2. Standardize the input features using scaling.
3. Train two SGDRegressor models: one for predicting price and one for occupants.
4. Take user input, scale it, and use both models to predict and display results.

## Program:
```
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("house.csv")
#print(data.columns)
data.columns = data.columns.str.strip()
X = data[['Size', 'Bedrooms']]
y_price = data['Price']
y_occ = data['Occupants']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)
size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))
new_data = scaler.transform([[size, bed]])
pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: 
RegisterNumber:  
*/
```

## Output:
<img width="1124" height="89" alt="image" src="https://github.com/user-attachments/assets/9cf4dae9-4fc2-4b03-aff7-a3e51f62444a" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
