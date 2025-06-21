import wfdb
import matplotlib.pyplot as plt
import os

# === SETTINGS ===
record_name = 'a01'  # You can change this to 'a02', 'x35', etc.

# === Load the record and annotation ===
record = wfdb.rdrecord(record_name)
annotation = wfdb.rdann(record_name, 'apn')

# === Print signal info ===
print(f"\nLoaded record: {record_name}")
print("Signal shape:", record.p_signal.shape)
print("Sampling frequency (Hz):", record.fs)
print("Annotation positions (samples):", annotation.sample[:10])
print("Annotation symbols (apnea = 'A'):", annotation.symbol[:10])

# === Plot the first 10 seconds of ECG ===
fs = record.fs
signal = record.p_signal[:, 0]  # single channel ECG
time = [i/fs for i in range(len(signal))]

plt.figure(figsize=(12, 4))
plt.plot(time[:6000], signal[:6000])
plt.title(f"ECG Signal - Record {record_name} (First 10 sec)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
