import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# --- 1. Data Loading & Preprocessing ---
# Dummy dataset creation for illustration
data = {
    "order_id": range(1, 101),
    "delivery_lat": np.random.uniform(28.5, 28.7, 100),
    "delivery_lng": np.random.uniform(77.1, 77.3, 100),
    "distance_km": np.random.uniform(2.0, 25.0, 100),
    "package_weight_kg": np.random.uniform(0.5, 15.0, 100),
    "hour_of_day": np.random.randint(8, 22, 100),
    "is_weekend": np.random.choice([0, 1], 100),
    "traffic_index": np.random.uniform(1.0, 5.0, 100),
    "delivery_time_mins": np.random.uniform(15.0, 120.0, 100),
}
df = pd.DataFrame(data)

# Impute missing values if any
df["package_weight_kg"].fillna(df["package_weight_kg"].median(), inplace=True)

# --- 2. Spatial Clustering of Delivery Zones ---
coordinates = df[["delivery_lat", "delivery_lng"]]
kmeans = KMeans(n_clusters=5, random_state=42)
df["delivery_cluster"] = kmeans.fit_predict(coordinates)

print("Cluster Centers (Micro-Hub Locations):")
print(kmeans.cluster_centers_)

# --- 3. Predictive Modeling for Delivery Duration ---
features = [
    "distance_km",
    "package_weight_kg",
    "hour_of_day",
    "is_weekend",
    "traffic_index",
]
X = df[features]
y = df["delivery_time_mins"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error in Delivery Time Prediction: {mae:.2f} minutes")
