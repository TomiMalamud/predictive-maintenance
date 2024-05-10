import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Step 1: Simulate input data
np.random.seed(42)
timestamps = pd.date_range(start="2024-01-01 08:00:00", periods=10000, freq="min")
cycle_time = np.random.normal(loc=0.7, scale=0.05, size=10000)
temperature = np.random.normal(loc=75, scale=3, size=10000)
vibration = np.random.normal(loc=0.02, scale=0.005, size=10000)
error_types = np.random.choice(['APP', 'APE', 'APF'], size=10000, p=[0.4, 0.3, 0.3])
stop_event = np.random.choice([0, 1], size=10000, p=[0.9, 0.1])

# Create a DataFrame
data = pd.DataFrame({
    'timestamp': timestamps,
    'cycle_time': cycle_time,
    'temperature': temperature,
    'vibration': vibration,
    'error_type': error_types,
    'stop_event': stop_event
})

# Step 2: Feature Engineering
# Rolling features
data['cycle_time_avg_10m'] = data['cycle_time'].rolling(window=10, min_periods=1).mean()
data['temperature_avg_10m'] = data['temperature'].rolling(window=10, min_periods=1).mean()
data['vibration_avg_10m'] = data['vibration'].rolling(window=10, min_periods=1).mean()

# One-hot encode error types
data = pd.get_dummies(data, columns=['error_type'], drop_first=True)

# Ensure all expected columns are present
expected_columns = ['cycle_time_avg_10m', 'temperature_avg_10m', 'vibration_avg_10m',
                    'error_type_APE', 'error_type_APF']
data = data.reindex(columns=expected_columns + ['stop_event'], fill_value=0)

# Final dataset
X = data[expected_columns]
y = data['stop_event']

# Step 3: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Adjust class weights in Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 5})
model.fit(X_train, y_train)

# Evaluate the model again
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Prediction Example
sample_data = pd.DataFrame([{
    'cycle_time_avg_10m': 0.72,
    'temperature_avg_10m': 76,
    'vibration_avg_10m': 0.021,
    'error_type_APE': 1,
    'error_type_APF': 0
}])
sample_data = sample_data.reindex(columns=expected_columns, fill_value=0)
prediction = model.predict(sample_data)
print(f'Predicted stop event: {prediction[0]}')
