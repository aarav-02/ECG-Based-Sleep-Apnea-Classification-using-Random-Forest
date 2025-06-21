import wfdb
import numpy as np
import pandas as pd
import os
from scipy.stats import entropy

# === SETTINGS ===
fs = 100  # 100 samples per second
samples_per_minute = fs * 60

# === GET ALL RECORDS ===
records = [f.split('.')[0] for f in os.listdir() if f.endswith('.hea')]
records = sorted(list(set(records)))  # remove duplicates

all_features = []

for record_name in records:
    try:
        print(f"Processing: {record_name}")
        record = wfdb.rdrecord(record_name)
        annotation = wfdb.rdann(record_name, 'apn')
        signal = record.p_signal[:, 0]
        
        # Labels: 1 per minute (A = Apnea, N = Normal)
        labels = [1 if sym == 'A' else 0 for sym in annotation.symbol]
        num_minutes = len(labels)

        for i in range(num_minutes):
            start = i * samples_per_minute
            end = start + samples_per_minute
            if end > len(signal):
                break

            segment = signal[start:end]

            # === Feature extraction ===
            mean_val = np.mean(segment)
            std_val = np.std(segment)
            energy = np.sum(segment**2)
            zero_crossings = np.sum(np.diff(np.sign(segment)) != 0)
            
            # === Additional Features ===
            # Heart rate (based on RR intervals)
            rr_intervals = np.diff(np.where(np.diff(np.sign(segment)) != 0)[0])
            heart_rate = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0

            # Entropy (signal complexity)
            signal_entropy = entropy(np.histogram(segment, bins=10)[0])

            all_features.append([record_name, i, mean_val, std_val, energy, zero_crossings, heart_rate, signal_entropy, labels[i]])

    except Exception as e:
        print(f"❌ Skipping {record_name} due to error: {e}")

# === SAVE TO CSV ===
df = pd.DataFrame(all_features, columns=[
    'record', 'minute', 'mean', 'std', 'energy', 'zero_crossings', 'heart_rate', 'entropy', 'label'
])
df.to_csv('all_features_improved.csv', index=False)
print("\n✅ All improved features saved to all_features_improved.csv")