import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Dataset Simulation
np.random.seed(42)
n_samples = 500

routes = ['Route A (North)', 'Route B (South)', 'Route C (East)', 'Route D (West)']
carriers = ['Express Shipping Co.', 'Standard Freight', 'Rapid Delivery', 'LogiTrans Services']

data = {
    'Shipment_ID': [f'SHP-{1000+i}' for i in range(n_samples)],
    'Route': np.random.choice(routes, n_samples, p=[0.3, 0.25, 0.25, 0.2]),
    'Carrier': np.random.choice(carriers, n_samples, p=[0.25, 0.35, 0.2, 0.2]),
    'Distance_km': np.random.uniform(50, 1200, n_samples).round(1),
    'Weight_kg': np.random.uniform(5, 500, n_samples).round(1),
    'Shipment_Volume_units': np.random.randint(10, 200, n_samples),
    'Delivery_Time_days': np.random.exponential(scale=2.5, size=n_samples).round(1) + 1,
    'Delay_Status': np.random.choice(['On Time', 'Delayed', 'Severely Delayed'], n_samples, p=[0.7, 0.2, 0.1])
}

df = pd.DataFrame(data)
carrier_mult = {'Express Shipping Co.': 1.5, 'Standard Freight': 1.0, 'Rapid Delivery': 1.3, 'LogiTrans Services': 0.9}
df['Transportation_Cost_USD'] = ((df['Distance_km'] * 0.8 + df['Weight_kg'] * 1.2) * 
                                  df['Carrier'].map(carrier_mult) + 
                                  np.random.normal(50, 15, n_samples)).round(2)

# Save simulated dataset
df.to_csv('logistics_dataset.csv', index=False)

# 2. Visualizations
plt.style.use('seaborn-v0_8-whitegrid')

# Fig 1: Transit Times Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df['Delivery_Time_days'], kde=True, color='#1F4E78', bins=25)
plt.title('Distribution of Delivery Transit Times (Days)', fontweight='bold')
plt.xlabel('Delivery Time (Days)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('delivery_time_distribution.png', dpi=300)
plt.close()

# Fig 2: Cost vs Weight
plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x='Weight_kg', y='Transportation_Cost_USD', hue='Carrier', alpha=0.7, palette='tab10')
plt.title('Transportation Cost vs. Shipment Weight by Carrier', fontweight='bold')
plt.xlabel('Weight (kg)')
plt.ylabel('Transportation Cost ($)')
plt.tight_layout()
plt.savefig('cost_vs_weight.png', dpi=300)
plt.close()

# Fig 3: Cost by Carrier Boxplot
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x='Carrier', y='Transportation_Cost_USD', palette='Set2')
plt.title('Transportation Cost Variance Across Logistics Carriers', fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('cost_by_carrier.png', dpi=300)
plt.close()

# Fig 4: Heatmap
plt.figure(figsize=(6, 4.5))
numeric_df = df[['Distance_km', 'Weight_kg', 'Shipment_Volume_units', 'Delivery_Time_days', 'Transportation_Cost_USD']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Correlation Matrix of Key Logistics Variables', fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()

print("Analysis and visualizations completed successfully.")
