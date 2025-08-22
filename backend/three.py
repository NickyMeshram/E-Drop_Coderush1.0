import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import firebase_admin
from firebase_admin import credentials, db
import joblib

# --- Step 1: Load and Clean Data ---
df = pd.read_csv("E:/CodeRush/cme_demo_dataset_v2.csv", parse_dates=['timestamp'])

# Normalize column names
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(r'[^\w\s]', '', regex=True)
              .str.replace(r'\s+', '_', regex=True)
)

print("Cleaned columns:", df.columns.tolist())

# --- Step 2: Feature Engineering ---
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day

# Ensure magnetic field column exists
if 'magnetic_field_nt' not in df.columns:
    print("'magnetic_field_nt' column missing. Please verify dataset.")
    df['magnetic_field_nt'] = 0  # fallback default

# --- Step 3: Auto-label CME Events ---
df['cme_event'] = (
    (df['speed_kms'] > 400) &
    (df['kinetic_energy_j'] > 1e23)
).astype(int)

# --- Step 4: Define Features ---
features = [
    'speed_kms', 'acceleration_ms2', 'angular_width', 'direction_pa',
    'mass_kg', 'kinetic_energy_j', 'brightness', 'magnetic_field_nt',
    'solar_wind_speed_kms', 'solar_wind_density_particles_cm3',
    'hour', 'day'
]

X = df[features].fillna(0)
y = df['cme_event']

# --- Step 5: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 6: Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 7: Evaluate Model ---
y_pred = model.predict(X_test)
print("\n Classification Report:")
print(classification_report(y_test, y_pred))
print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Step 8: Save Model ---
joblib.dump(model, "E:/CodeRush/cme_model_firebase.pkl")
print("\n Model saved as 'cme_model_firebase.pkl'")

# --- Step 9: Predict New CME Alerts ---
new_data = df.tail(100)[features].fillna(0)
predictions = model.predict(new_data)

alert_df = df.tail(100).copy()
alert_df['predicted_cme'] = predictions
cme_alerts = alert_df[alert_df['predicted_cme'] == 1]

print(f"\n Predicted CME alerts: {len(cme_alerts)}")
print(cme_alerts[['timestamp', 'speed_kms', 'kinetic_energy_j']].head())

# --- Step 10: Send Alerts to Firebase ---
if not firebase_admin._apps:
    cred = credentials.Certificate(r"E:\CodeRush\fir-realtimekotlin-470d2-firebase-adminsdk-nzy4k-fd6747bd4a.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://fir-realtimekotlin-470d2-default-rtdb.firebaseio.com/'
    })

ref = db.reference('cme_alerts')
for _, row in cme_alerts.iterrows():
    alert = {
        'timestamp': str(row['timestamp']),
        'speed': row['speed_kms'],
        'kinetic_energy': row['kinetic_energy_j'],
        'magnetic_field': row['magnetic_field_nt']
    }
    ref.push(alert)

print("\n CME alerts sent to Firebase.")
