import wfdb
import numpy as np
import joblib
import pandas as pd
from scipy.stats import entropy

# === Load trained model ===
model = joblib.load('apnea_model_improved.pkl')

# === ECG record to test ===
record_name = 'a09'  
fs = 100
samples_per_minute = fs * 60

# === Load ECG signal ===
record = wfdb.rdrecord(record_name)
signal = record.p_signal[:, 0]

# === Break into 1-minute chunks and predict ===
num_chunks = len(signal) // samples_per_minute

for i in range(num_chunks):
    start = i * samples_per_minute
    end = start + samples_per_minute
    segment = signal[start:end]

    # === Extract features ===
    mean_val = np.mean(segment)
    std_val = np.std(segment)
    energy = np.sum(segment**2)
    zero_crossings = np.sum(np.diff(np.sign(segment)) != 0)

    # Heart rate (based on RR intervals)
    rr_intervals = np.diff(np.where(np.diff(np.sign(segment)) != 0)[0])
    heart_rate = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0

    # Entropy (signal complexity)
    signal_entropy = entropy(np.histogram(segment, bins=10)[0])

    # Create a DataFrame with feature names
    features = pd.DataFrame([[mean_val, std_val, energy, zero_crossings, heart_rate, signal_entropy]], 
                            columns=['mean', 'std', 'energy', 'zero_crossings', 'heart_rate', 'entropy'])

    # Predict apnea or normal
    prediction = model.predict(features)

    label = "Apnea 😴" if prediction[0] == 1 else "Normal 🟢"
    print(f"Minute {i:02d}: {label}")
