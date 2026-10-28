import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Dataset Simulation
np.random.seed(42)
n_samples = 600

routes = ['Route A (North)', 'Route B (South)', 'Route C (East)', 'Route D (West)']
carriers = ['Express Shipping Co.', 'Standard Freight', 'Rapid Delivery', 'LogiTrans Services']

distance = np.random.uniform(50, 1200, n_samples)
weight = np.random.uniform(5, 500, n_samples)
volume = np.random.randint(10, 200, n_samples)
route_choice = np.random.choice(routes, n_samples, p=[0.3, 0.25, 0.25, 0.2])
carrier_choice = np.random.choice(carriers, n_samples, p=[0.25, 0.35, 0.2, 0.2])

carrier_speed_mult = {'Express Shipping Co.': 0.7, 'Standard Freight': 1.1, 'Rapid Delivery': 0.85, 'LogiTrans Services': 1.0}
route_traffic_mult = {'Route A (North)': 1.3, 'Route B (South)': 1.0, 'Route C (East)': 1.1, 'Route D (West)': 0.9}

carrier_mult_arr = np.array([carrier_speed_mult[c] for c in carrier_choice])
route_mult_arr = np.array([route_traffic_mult[r] for r in route_choice])

base_time = (distance / 250.0) * route_mult_arr * carrier_mult_arr + (weight / 300.0)
delivery_time = np.maximum(1.0, base_time + np.random.normal(0, 0.4, n_samples)).round(2)

df = pd.DataFrame({
    'Distance_km': distance.round(1),
    'Weight_kg': weight.round(1),
    'Shipment_Volume_units': volume,
    'Route': route_choice,
    'Carrier': carrier_choice,
    'Delivery_Time_days': delivery_time
})

# 2. Predictive Modeling
df_encoded = pd.get_dummies(df, columns=['Route', 'Carrier'], drop_first=True)
X = df_encoded.drop(columns=['Delivery_Time_days'])
y = df_encoded['Delivery_Time_days']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Random Forest Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Random Forest Test R2:", r2_score(y_test, y_pred))
