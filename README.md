# ECG-Based-Sleep-Apnea-Classification-using-Random-Forest

🧠 Project Idea
Sleep Apnea is a serious disorder where breathing repeatedly stops during sleep.
Early detection can help prevent complications like heart disease, stroke, etc.
Goal: Detect apnea episodes from ECG signals using machine learning.

🎯 Project Objectives
Collect raw ECG data from PhysioNet Apnea-ECG database.
Extract meaningful features from minute-wise ECG signals.
Train a classification model to predict apnea events.
Achieve high accuracy (82% achieved!) with robust validation.

🔥 Key Achievements
📈 Accuracy achieved: 82%

🔍 Extracted 7+ important features (Mean, Std, Energy, Heart Rate, Entropy etc.).

🤖 Machine Learning Model: XGBoost Classifier

🛠 Techniques used: Leave-One-Patient-Out Cross Validation (LOPO-CV) for better generalization.

🛠️ Technologies Used
Python 3.12
Libraries: Pandas, Numpy, WFDB, XGBoost, Scikit-learn, Scipy
Dataset: PhysioNet Apnea-ECG Database
Feature Extraction: Custom Signal Processing
Validation Strategy: LOPO Cross-Validation

🧩 Feature Importance
Feature	Importance
Energy	High
Entropy	High
Heart Rate	Moderate
Zero Crossings	Moderate
Mean & Std	Supportive

🚀 Future Scope
Add more complex ECG features (like spectral analysis).
Train Deep Learning models (CNNs or LSTMs).
Apply on larger datasets.
Create a mobile app for apnea monitoring.

