import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# === Load the improved dataset ===
df = pd.read_csv('all_features_improved.csv')

# === Features and labels ===
X = df[['mean', 'std', 'energy', 'zero_crossings', 'heart_rate', 'entropy']]
y = df['label']

# === Split into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Hyperparameter tuning using GridSearchCV ===
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Initialize RandomForest and GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Train the model with GridSearch
grid_search.fit(X_train, y_train)

# Get the best parameters
print("\nBest Parameters found: ", grid_search.best_params_)

# === Use the best model from GridSearch ===
best_model = grid_search.best_estimator_

# === Test the best model ===
y_pred = best_model.predict(X_test)

# === Check accuracy ===
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# === Additional Evaluation Metrics ===
print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# === Optional: Save the tuned model ===
import joblib
joblib.dump(best_model, 'apnea_model_tuned.pkl')
print("\n✅ Tuned model saved to apnea_model_tuned.pkl")