import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Step 1: Load the Saved Model ---
model = joblib.load("E:/CodeRush/cme_model_firebase.pkl")
print("Model loaded successfully.")

# --- Step 2: Load New Test Data ---
test_df = pd.read_csv("cme_test.csv", parse_dates=['timestamp'])

# Normalize column names
test_df.columns = (
    test_df.columns.str.strip()
                   .str.lower()
                   .str.replace(r'[^\w\s]', '', regex=True)
                   .str.replace(r'\s+', '_', regex=True)
)

# --- Step 3: Feature Engineering ---
test_df['hour'] = test_df['timestamp'].dt.hour
test_df['day'] = test_df['timestamp'].dt.day

if 'magnetic_field_nt' not in test_df.columns:
    print("'magnetic_field_nt' column missing. Using default value 0.")
    test_df['magnetic_field_nt'] = 0

# --- Step 4: Auto-label CME Events (if needed for evaluation) ---
test_df['cme_event'] = (
    (test_df['speed_kms'] > 400) &
    (test_df['kinetic_energy_j'] > 1e23)
).astype(int)

# --- Step 5: Define Features and Labels ---
features = [
    'speed_kms', 'acceleration_ms2', 'angular_width', 'direction_pa',
    'mass_kg', 'kinetic_energy_j', 'brightness', 'magnetic_field_nt',
    'solar_wind_speed_kms', 'solar_wind_density_particles_cm3',
    'hour', 'day'
]

X_test = test_df[features].fillna(0)
y_true = test_df['cme_event']

# --- Step 6: Make Predictions ---
y_pred = model.predict(X_test)

# --- Step 7: Evaluate ---
print("\n Classification Report:")
print(classification_report(y_true, y_pred))

print("\n Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

accuracy = accuracy_score(y_true, y_pred)
print(f"\n Accuracy: {accuracy:.4f}")
