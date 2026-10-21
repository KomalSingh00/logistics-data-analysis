import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Simulate Logistics Dataset Creation (150,000 rows)
np.random.seed(42)
n_samples = 150000

data = {
    'Shipment_ID': [f'SHP_{i:06d}' for i in range(n_samples)],
    'Transport_Mode': np.random.choice(
        ['Ocean', 'Air', 'Road', 'Rail'], size=n_samples, p=[0.5, 0.1, 0.3, 0.1]
    ),
    'Planned_Lead_Time_Days': np.random.randint(1, 45, size=n_samples),
    'Actual_Lead_Time_Days': np.random.normal(loc=12, scale=5, size=n_samples),
    'Container_Weight_Tons': np.random.uniform(1.0, 30.0, size=n_samples),
    'Shipping_Cost_USD': np.random.exponential(scale=1200, size=n_samples)
    + 200,
    'Warehouse_Delay_Hours': np.random.exponential(scale=5, size=n_samples),
}

df = pd.DataFrame(data)

# Inject synthetic missing values and outliers for cleaning demonstration
df.loc[df.sample(frac=0.034).index, 'Actual_Lead_Time_Days'] = np.nan
df.loc[df.sample(frac=0.012).index, 'Shipping_Cost_USD'] = np.nan
df.loc[0:15, 'Actual_Lead_Time_Days'] = -10.0  # Erroneous negative lead time
df.loc[16:30, 'Shipping_Cost_USD'] = 95000.0  # Extreme anomaly outlier


# 2. Pipeline Stage A: Cleaning & Logical Bounding
def clean_logistics_data(data_frame):
  df_clean = data_frame.copy()

  # Remove logical impossibilities (negative lead times)
  df_clean.loc[df_clean['Actual_Lead_Time_Days'] <= 0, 'Actual_Lead_Time_Days'] = (
      np.nan
  )

  # Grouped Median Imputation for Actual_Lead_Time_Days
  df_clean['Actual_Lead_Time_Days'] = df_clean.groupby('Transport_Mode')[
      'Actual_Lead_Time_Days'
  ].transform(lambda x: x.fillna(x.median()))

  # MICE Imputation for Shipping_Cost_USD
  mice_imputer = IterativeImputer(max_iter=10, random_state=42)
  num_cols_for_mice = [
      'Planned_Lead_Time_Days',
      'Container_Weight_Tons',
      'Shipping_Cost_USD',
  ]
  df_clean[num_cols_for_mice] = mice_imputer.fit_transform(
      df_clean[num_cols_for_mice]
  )

  return df_clean


df_cleaned = clean_logistics_data(df)


# 3. Pipeline Stage B: IQR Outlier Trimming
def remove_iqr_outliers(df_in, column, multiplier=1.5):
  q1 = df_in[column].quantile(0.25)
  q3 = df_in[column].quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - (multiplier * iqr)
  upper_bound = q3 + (multiplier * iqr)

  # Cap extreme outliers to upper/lower bounds
  df_in[column] = np.clip(df_in[column], lower_bound, upper_bound)
  return df_in


df_cleaned = remove_iqr_outliers(
    df_cleaned, 'Shipping_Cost_USD', multiplier=2.5
)


# 4. Pipeline Stage C: Normalization & Scaling
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

df_cleaned['Scaled_Container_Weight'] = scaler_standard.fit_transform(
    df_cleaned[['Container_Weight_Tons']]
)
df_cleaned['Normalized_Shipping_Cost'] = scaler_minmax.fit_transform(
    df_cleaned[['Shipping_Cost_USD']]
)

print('Preprocessing Pipeline Completed Successfully.')
print(f'Final Cleaned Dataset Shape: {df_cleaned.shape}')
print(f'Remaining Missing Values: {df_cleaned.isnull().sum().sum()}')
