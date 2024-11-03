
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


data = pd.read_csv(r"C:\Users\User\Desktop\ML\dataset.csv")

print(data.head())


# Supprimer les valeurs manquantes
data = data.dropna()

# Convertir les colonnes en numériques si nécessaire
data['Bedrooms'] = pd.to_numeric(data['Bedrooms'], errors='coerce')
data['Bathrooms'] = pd.to_numeric(data['Bathrooms'], errors='coerce')
data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
data['Floors'] = pd.to_numeric(data['Floors'], errors='coerce')
data['YearBuilt'] = pd.to_numeric(data['YearBuilt'], errors='coerce')
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

# Supprimer les lignes avec des valeurs non numériques après conversion
data = data.dropna()


X = data[['Bedrooms', 'Bathrooms', 'Area', 'Floors', 'YearBuilt']]  # features
y = data['Price']  #target

# 5. Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)






svr_linear = SVR(kernel='linear')
svr_linear.fit(X_train, y_train)
y_pred_svr_linear = svr_linear.predict(X_test)


svr_poly = SVR(kernel='poly', degree=3)
svr_poly.fit(X_train, y_train)
y_pred_svr_poly = svr_poly.predict(X_test)


svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(X_train, y_train)
y_pred_svr_rbf = svr_rbf.predict(X_test)


svr_sigmoid = SVR(kernel='sigmoid')
svr_sigmoid.fit(X_train, y_train)
y_pred_svr_sigmoid = svr_sigmoid.predict(X_test)



print("\nSVR linéaire:")
print("MSE: ", mean_squared_error(y_test, y_pred_svr_linear))
print("R²: ", r2_score(y_test, y_pred_svr_linear))

print("\nSVR polynômial:")
print("MSE: ", mean_squared_error(y_test, y_pred_svr_poly))
print("R²: ", r2_score(y_test, y_pred_svr_poly))

print("\nSVR RBF:")
print("MSE: ", mean_squared_error(y_test, y_pred_svr_rbf))
print("R²: ", r2_score(y_test, y_pred_svr_rbf))

print("\nSVR Sigmoid:")
print("MSE: ", mean_squared_error(y_test, y_pred_svr_sigmoid))
print("R²: ", r2_score(y_test, y_pred_svr_sigmoid))



